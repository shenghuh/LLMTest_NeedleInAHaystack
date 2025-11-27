import os
import json
import torch
from dataclasses import dataclass, field
from typing import Optional
from transformers import AutoModelForCausalLM, AutoTokenizer
from datetime import datetime


@dataclass
class BenchmarkConfig:
    model_name: str = "meta-llama/Llama-2-7b-chat-hf"
    haystack_path: str = "../PaulGrahamEssays"
    needle: str = "The best thing to do in San Francisco is eat a sandwich and sit in Dolores Park on a sunny day."
    retrieval_question: str = "What is the best thing to do in San Francisco?"
    context_lengths: list = field(default_factory=lambda: [1000, 2000, 4000])
    needle_depths: list = field(default_factory=lambda: [0.0, 0.5, 1.0])
    output_dir: str = "./results"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    max_new_tokens: int = 256
    torch_dtype: str = "float16"
    use_llm_judge: bool = True  # New: toggle LLM-based evaluation


class HFBenchmarker:
    def __init__(self, config: BenchmarkConfig):
        self.config = config
        self.model = None
        self.tokenizer = None
        self.haystack_text = ""
        self._load_model()
        self._load_haystack()

    def _load_model(self):
        print(f"Loading model: {self.config.model_name}")
        dtype = getattr(torch, self.config.torch_dtype)
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.model_name,
            torch_dtype=dtype,
            device_map="auto",
            trust_remote_code=True
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

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

    def _build_context(self, context_length: int, needle_depth: float) -> str:
        """Build context with needle inserted at specified depth."""
        tokens = self.tokenizer.encode(self.haystack_text, add_special_tokens=False)
        target_tokens = context_length - len(self.tokenizer.encode(self.config.needle, add_special_tokens=False)) - 50
        tokens = tokens[:max(target_tokens, 100)]
        text = self.tokenizer.decode(tokens, skip_special_tokens=True)
        
        # Insert needle at depth
        insert_pos = int(len(text) * needle_depth)
        context = text[:insert_pos] + f"\n\n{self.config.needle}\n\n" + text[insert_pos:]
        return context

    def _build_prompt(self, context: str) -> str:
        """Build the full prompt with context and question."""
        return (
            f"<s>[INST] You are a helpful assistant. Read the following context and answer the question.\n\n"
            f"Context:\n{context}\n\n"
            f"Question: {self.config.retrieval_question}\n\n"
            f"Answer concisely based only on the context provided. [/INST]"
        )

    def _generate(self, prompt: str) -> str:
        """Generate response from the model."""
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True).to(self.model.device)
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.config.max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id
            )
        response = self.tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
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
Score 5: The response captures some elements of the needle but is incomplete or partially incorrect.
Score 7: The response is mostly accurate and captures the main idea of the needle with minor omissions.
Score 8: The response is accurate and includes most key details from the needle.
Score 9: The response is highly accurate, comprehensive, and directly addresses the question using the needle.
Score 10: The response perfectly retrieves and presents the needle information, fully answering the question.

Only respond with a JSON object in this format:
{{"score": <integer 1-10>, "reasoning": "<brief explanation>"}}
[/INST]"""

        inputs = self.tokenizer(judge_prompt, return_tensors="pt", truncation=True).to(self.model.device)
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=150,
                do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id
            )
        judge_response = self.tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
        
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
                
                context = self._build_context(ctx_len, depth)
                prompt = self._build_prompt(context)
                
                start_time = datetime.now()
                response = self._generate(prompt)
                latency = (datetime.now() - start_time).total_seconds()
                
                eval_result = self._evaluate(response)
                
                result = {
                    "context_length": ctx_len,
                    "needle_depth": depth,
                    "response": response,
                    "latency_seconds": latency,
                    **eval_result
                }
                results.append(result)
                
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
    parser.add_argument("--haystack", type=str, default="../PaulGrahamEssays")
    parser.add_argument("--context-lengths", type=int, nargs="+", default=[1000, 2000, 4000])
    parser.add_argument("--depths", type=float, nargs="+", default=[0.0, 0.5, 1.0])
    parser.add_argument("--output-dir", type=str, default="./results")
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--no-llm-judge", action="store_true", help="Use keyword matching instead of LLM judge")
    args = parser.parse_args()
    
    config = BenchmarkConfig(
        model_name=args.model,
        haystack_path=args.haystack,
        context_lengths=args.context_lengths,
        needle_depths=args.depths,
        output_dir=args.output_dir,
        max_new_tokens=args.max_tokens,
        use_llm_judge=not args.no_llm_judge
    )
    
    benchmarker = HFBenchmarker(config)
    benchmarker.run()


if __name__ == "__main__":
    main()
