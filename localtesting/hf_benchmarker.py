import os
import json
import torch
import torch.nn.functional as F
from dataclasses import dataclass, field
from typing import Optional
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from datetime import datetime

access_token = ""

@dataclass
class BenchmarkConfig:
    model_name: str = "meta-llama/Llama-2-7b-chat-hf"
    judge_model_name: str = None  # Separate judge model (None = use same model)
    encoder_model_name: str = None  # Encoder for filtering (None = no filtering)
    haystack_path: str = "../PaulGrahamEssays"
    needle: str = "The best thing to do in San Francisco is eat a sandwich and sit in Dolores Park on a sunny day."
    retrieval_question: str = "What is the best thing to do in San Francisco?"
    context_lengths: list = field(default_factory=lambda: [1000, 2000, 4000])
    needle_depths: list = field(default_factory=lambda: [0.0, 0.5, 1.0])
    output_dir: str = "./results"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    max_new_tokens: int = 256
    torch_dtype: str = "float16"
    use_llm_judge: bool = True
    skip_evaluation: bool = False
    # Quantization options
    quantization: str = None  # None, "4bit", or "8bit"
    # Encoder filtering options
    filter_chunk_size: int = 256  # Number of tokens per chunk
    filter_top_k: int = 5  # Number of top chunks to keep
    use_encoder_filter: bool = False  # Toggle encoder-based filtering


class HFBenchmarker:
    def __init__(self, config: BenchmarkConfig):
        self.config = config
        self.model = None
        self.tokenizer = None
        self.judge_model = None
        self.judge_tokenizer = None
        self.encoder_model = None
        self.haystack_text = ""
        self._load_model()
        self._load_judge_model()
        self._load_encoder_model()
        self._load_haystack()

    def _get_quantization_config(self):
        """Get BitsAndBytesConfig based on quantization setting."""
        if self.config.quantization == "4bit":
            return BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=getattr(torch, self.config.torch_dtype),
                bnb_4bit_use_double_quant=True,
            )
        elif self.config.quantization == "8bit":
            return BitsAndBytesConfig(
                load_in_8bit=True,
            )
        return None

    def _load_model(self):
        print(f"Loading model: {self.config.model_name}")
        if self.config.quantization:
            print(f"  Using {self.config.quantization} quantization")
        
        dtype = getattr(torch, self.config.torch_dtype)
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_name, token=access_token)
        
        quant_config = self._get_quantization_config()
        
        load_kwargs = {
            "device_map": "auto",
            "trust_remote_code": True,
            "token": access_token,
        }
        
        if quant_config:
            load_kwargs["quantization_config"] = quant_config
        else:
            load_kwargs["torch_dtype"] = dtype
        
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.model_name,
            **load_kwargs
        )
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def _load_judge_model(self):
        """Load separate judge model if specified, otherwise use the main model."""
        if self.config.skip_evaluation:
            self.judge_model = None
            self.judge_tokenizer = None
            return
            
        if self.config.judge_model_name and self.config.use_llm_judge:
            print(f"Loading judge model: {self.config.judge_model_name}")
            if self.config.quantization:
                print(f"  Using {self.config.quantization} quantization")
            
            dtype = getattr(torch, self.config.torch_dtype)
            self.judge_tokenizer = AutoTokenizer.from_pretrained(
                self.config.judge_model_name, 
                token=access_token
            )
            
            quant_config = self._get_quantization_config()
            
            load_kwargs = {
                "device_map": "auto",
                "trust_remote_code": True,
                "token": access_token,
            }
            
            if quant_config:
                load_kwargs["quantization_config"] = quant_config
            else:
                load_kwargs["torch_dtype"] = dtype
            
            self.judge_model = AutoModelForCausalLM.from_pretrained(
                self.config.judge_model_name,
                **load_kwargs
            )
            
            if self.judge_tokenizer.pad_token is None:
                self.judge_tokenizer.pad_token = self.judge_tokenizer.eos_token
        else:
            # Use the same model for judging
            self.judge_model = self.model
            self.judge_tokenizer = self.tokenizer

    def _load_encoder_model(self):
        """Load encoder model for context filtering if specified."""
        if self.config.encoder_model_name and self.config.use_encoder_filter:
            print(f"Loading encoder model: {self.config.encoder_model_name}")
            try:
                from sentence_transformers import SentenceTransformer
                self.encoder_model = SentenceTransformer(
                    self.config.encoder_model_name,
                    device=self.config.device,
                )
            except ImportError:
                print("Warning: sentence-transformers not installed. Install with: pip install sentence-transformers")
                self.encoder_model = None
                self.config.use_encoder_filter = False
        else:
            self.encoder_model = None

    def _load_haystack(self):
        texts = []
        if os.path.isdir(self.config.haystack_path):
            for fname in sorted(os.listdir(self.config.haystack_path)):
                fpath = os.path.join(self.config.haystack_path, fname)
                if os.path.isfile(fpath):
                    with open(fpath, "r", encoding="utf-8") as f:
                        texts.append(f.read())
        else:
            with open(self.config.haystack_path, "r", encoding="utf-8") as f:
                texts.append(f.read())
        self.haystack_text = "\n\n".join(texts)
        print(f"Loaded haystack with {len(self.haystack_text)} characters")

    def _split_into_chunks(self, text: str, chunk_size: int) -> list:
        """Split text into chunks of approximately chunk_size tokens."""
        tokens = self.tokenizer.encode(text, add_special_tokens=False)
        chunks = []
        
        for i in range(0, len(tokens), chunk_size):
            chunk_tokens = tokens[i:i + chunk_size]
            if chunk_tokens:
                chunk_text = self.tokenizer.decode(chunk_tokens, skip_special_tokens=True)
                chunks.append({
                    "text": chunk_text,
                    "start_idx": i,
                    "end_idx": i + len(chunk_tokens)
                })
        
        return chunks

    def _filter_context_with_encoder(self, context: str, question: str) -> str:
        """Filter context to keep only the most relevant chunks based on encoder similarity."""
        if not self.encoder_model:
            return context
        
        chunks = self._split_into_chunks(context, self.config.filter_chunk_size)
        
        if len(chunks) <= self.config.filter_top_k:
            return context  # No need to filter if fewer chunks than top_k
        
        # Encode question and chunks
        chunk_texts = [c["text"] for c in chunks]
        question_embedding = self.encoder_model.encode(question, convert_to_tensor=True)
        chunk_embeddings = self.encoder_model.encode(chunk_texts, convert_to_tensor=True)
        
        # Normalize embeddings
        question_embedding = F.normalize(question_embedding.unsqueeze(0), p=2, dim=1)
        chunk_embeddings = F.normalize(chunk_embeddings, p=2, dim=1)
        
        # Compute cosine similarity
        similarities = torch.matmul(chunk_embeddings, question_embedding.T).squeeze()
        
        # Get top-k chunk indices
        k = min(self.config.filter_top_k, len(chunks))
        topk_indices = torch.topk(similarities, k=k).indices.tolist()
        
        # Sort indices to maintain original order
        topk_indices = sorted(topk_indices)
        
        # Reconstruct filtered context
        selected_chunks = [chunks[i]["text"] for i in topk_indices]
        filtered_context = "\n\n".join(selected_chunks)
        
        return filtered_context

    def _build_context(self, context_length: int, needle_depth: float) -> tuple:
        """Build context with needle inserted at specified depth. Returns (full_context, filtered_context)."""
        tokens = self.tokenizer.encode(self.haystack_text, add_special_tokens=False)
        target_tokens = context_length - len(self.tokenizer.encode(self.config.needle, add_special_tokens=False)) - 50
        tokens = tokens[:max(target_tokens, 100)]
        text = self.tokenizer.decode(tokens, skip_special_tokens=True)
        
        # Insert needle at depth
        insert_pos = int(len(text) * needle_depth)
        full_context = text[:insert_pos] + f"\n\n{self.config.needle}\n\n" + text[insert_pos:]
        
        # Apply encoder filtering if enabled
        if self.config.use_encoder_filter and self.encoder_model:
            filtered_context = self._filter_context_with_encoder(
                full_context, 
                self.config.retrieval_question
            )
        else:
            filtered_context = full_context
        
        return full_context, filtered_context

    def _build_prompt(self, context: str) -> str:
        """Build the full prompt with context and question using the model's chat template."""
        messages = [
            {
                "role": "user",
                "content": (
                    f"You are a helpful assistant. Read the following context and answer the question.\n\n"
                    f"Context:\n{context}\n\n"
                    f"Question: {self.config.retrieval_question}\n\n"
                    f"Answer concisely based only on the context provided."
                )
            }
        ]
        
        # Use the tokenizer's chat template if available, otherwise fall back to basic format
        if hasattr(self.tokenizer, 'apply_chat_template') and self.tokenizer.chat_template:
            return self.tokenizer.apply_chat_template(
                messages, 
                tokenize=False, 
                add_generation_prompt=True
            )
        else:
            # Fallback for models without chat templates
            return (
                f"<s>[INST] {messages[0]['content']} [/INST]"
            )

    def _generate(self, prompt: str) -> str:
        """Generate response from the model."""
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True).to(self.model.device)
        
        # Debug: Print input info
        input_len = inputs["input_ids"].shape[1]
        print(f"  Input tokens: {input_len}")
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.config.max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id
            )
        
        # Debug: Print output info
        output_len = outputs.shape[1]
        new_tokens = output_len - input_len
        print(f"  Output tokens: {output_len} (generated {new_tokens} new tokens)")
        
        response = self.tokenizer.decode(outputs[0][input_len:], skip_special_tokens=True)
        
        # Debug: Print raw response if empty
        if not response.strip():
            raw_response = self.tokenizer.decode(outputs[0][input_len:], skip_special_tokens=False)
            print(f"  WARNING: Empty response. Raw tokens: {repr(raw_response[:100])}")
        
        return response.strip()

    def _evaluate_with_llm(self, response: str) -> dict:
        """Use the LLM to judge the response quality (NeedleInAHaystack criteria)."""
        judge_prompt = f"""<s>[INST] You are a helpful assistant evaluating responses for accuracy.

You are evaluating a response that was generated to answer a question based on a context that contained a specific piece of information (the "needle").

The needle (target information): "{self.config.needle}"

The question asked: "{self.config.retrieval_question}"

The model's response: "{response}"

Evaluate the response using the following criteria:

Score 1: The response is completely unrelated to the needle, or contradicts it entirely.
Score 2: The response mentions the topic but provides entirely wrong information.
Score 3: The response is partially related but misses all key details from the needle.
Score 4: The response is somewhat related and includes some details from the needle but lacks clarity or completeness.
Score 5: The response captures some elements of the needle but is incomplete or partially incorrect.
Score 7: The response is mostly accurate and captures the main idea of the needle with minor omissions.
Score 8: The response is accurate and includes most key details from the needle.
Score 9: The response is highly accurate, comprehensive, and directly addresses the question using the needle.
Score 10: The response perfectly retrieves and presents the needle information, fully answering the question.

Only respond with a JSON object in this format:
{{"score": <integer 1-10>, "reasoning": "<brief explanation>"}}
[/INST]"""

        inputs = self.judge_tokenizer(judge_prompt, return_tensors="pt", truncation=True).to(self.judge_model.device)
        with torch.no_grad():
            outputs = self.judge_model.generate(
                **inputs,
                max_new_tokens=150,
                do_sample=False,
                pad_token_id=self.judge_tokenizer.pad_token_id
            )
        judge_response = self.judge_tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
        
        # Parse the JSON response
        try:
            import re
            json_match = re.search(r'\{[^}]+\}', judge_response)
            if json_match:
                parsed = json.loads(json_match.group())
                score = int(parsed.get("score", 5))
                reasoning = parsed.get("reasoning", "")
            else:
                # Fallback: try to find a number
                numbers = re.findall(r'\b(10|[1-9])\b', judge_response)
                score = int(numbers[0]) if numbers else 5
                reasoning = judge_response
        except (json.JSONDecodeError, IndexError, ValueError):
            score = 5
            reasoning = f"Failed to parse judge response: {judge_response[:100]}"
        
        # Clamp score to valid range
        score = min(max(score, 1), 10)
        
        return {
            "score": score,
            "score_normalized": score / 10.0,
            "llm_reasoning": reasoning,
            "llm_raw_response": judge_response,
            "contains_answer": score >= 7  # Threshold for "successful" retrieval
        }

    def _evaluate(self, response: str) -> dict:
        """Evaluate response using LLM judge or keyword matching."""
        if self.config.skip_evaluation:
            return {
                "score": None,
                "score_normalized": None,
                "evaluation_skipped": True
            }
        
        if self.config.use_llm_judge:
            return self._evaluate_with_llm(response)
        
        # Fallback: keyword-based evaluation
        response_lower = response.lower()
        keywords = ["san francisco", "sandwich", "dolores park", "sunny day"]
        matches = sum(1 for kw in keywords if kw in response_lower)
        score = matches / len(keywords)
        
        return {
            "score": score,
            "matches": matches,
            "total_keywords": len(keywords),
            "contains_answer": score >= 0.5
        }

    def run(self) -> list:
        """Run the full benchmark."""
        results = []
        os.makedirs(self.config.output_dir, exist_ok=True)
        
        total_runs = len(self.config.context_lengths) * len(self.config.needle_depths)
        run_idx = 0
        
        for ctx_len in self.config.context_lengths:
            for depth in self.config.needle_depths:
                run_idx += 1
                print(f"\n[{run_idx}/{total_runs}] Context: {ctx_len} tokens, Depth: {depth:.0%}")
                
                full_context, filtered_context = self._build_context(ctx_len, depth)
                prompt = self._build_prompt(filtered_context)
                
                # Calculate context stats
                full_tokens = len(self.tokenizer.encode(full_context, add_special_tokens=False))
                filtered_tokens = len(self.tokenizer.encode(filtered_context, add_special_tokens=False))
                
                if self.config.use_encoder_filter:
                    print(f"  Context: {full_tokens} -> {filtered_tokens} tokens (filtered)")
                
                start_time = datetime.now()
                response = self._generate(prompt)
                latency = (datetime.now() - start_time).total_seconds()
                
                eval_result = self._evaluate(response)
                
                result = {
                    "context_length": ctx_len,
                    "needle_depth": depth,
                    "response": response,
                    "latency_seconds": latency,
                    "full_context_tokens": full_tokens,
                    "filtered_context_tokens": filtered_tokens,
                    "encoder_filtered": self.config.use_encoder_filter,
                    **eval_result
                }
                results.append(result)
                
                if self.config.skip_evaluation:
                    print(f"  Evaluation: skipped | Latency: {latency:.2f}s")
                else:
                    score_display = eval_result.get('score', eval_result.get('score_normalized', 0) * 10)
                    print(f"  Score: {score_display}/10 | Latency: {latency:.2f}s")
                print(f"  Response: {response[:100]}...")
        
        # Save results
        output_path = os.path.join(
            self.config.output_dir,
            f"results_{self.config.model_name.replace('/', '_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        )
        with open(output_path, "w") as f:
            json.dump({"config": self.config.__dict__, "results": results}, f, indent=2)
        print(f"\nResults saved to: {output_path}")
        
        return results


def main():
    import argparse
    parser = argparse.ArgumentParser(description="HuggingFace Needle-in-Haystack Benchmark")
    parser.add_argument("--model", type=str, default="meta-llama/Llama-2-7b-chat-hf")
    parser.add_argument("--judge-model", type=str, default=None, help="Separate model for judging (default: use same model)")
    parser.add_argument("--encoder-model", type=str, default=None, help="Encoder model for context filtering (e.g., sentence-transformers/all-MiniLM-L6-v2)")
    parser.add_argument("--haystack", type=str, default="../PaulGrahamEssays")
    parser.add_argument("--context-lengths", type=int, nargs="+", default=[1000, 2000, 4000])
    parser.add_argument("--depths", type=float, nargs="+", default=[0.0, 0.5, 1.0])
    parser.add_argument("--output-dir", type=str, default="./results")
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--no-llm-judge", action="store_true", help="Use keyword matching instead of LLM judge")
    parser.add_argument("--skip-evaluation", action="store_true", help="Skip evaluation entirely (only generate responses)")
    parser.add_argument("--quantization", type=str, choices=["4bit", "8bit"], default=None, help="Quantization mode for reduced memory usage")
    parser.add_argument("--filter-chunk-size", type=int, default=256, help="Chunk size in tokens for encoder filtering")
    parser.add_argument("--filter-top-k", type=int, default=5, help="Number of top chunks to keep after filtering")
    args = parser.parse_args()
    
    config = BenchmarkConfig(
        model_name=args.model,
        judge_model_name=args.judge_model,
        encoder_model_name=args.encoder_model,
        haystack_path=args.haystack,
        context_lengths=args.context_lengths,
        needle_depths=args.depths,
        output_dir=args.output_dir,
        max_new_tokens=args.max_tokens,
        use_llm_judge=not args.no_llm_judge,
        skip_evaluation=args.skip_evaluation,
        quantization=args.quantization,
        use_encoder_filter=args.encoder_model is not None,
        filter_chunk_size=args.filter_chunk_size,
        filter_top_k=args.filter_top_k
    )
    
    benchmarker = HFBenchmarker(config)
    benchmarker.run()


if __name__ == "__main__":
    main()
