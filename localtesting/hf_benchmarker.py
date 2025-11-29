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
    filter_chunk_stride: int = None  # Stride between chunks (None = same as chunk_size, i.e., no overlap)
    filter_top_k: int = 5  # Number of top chunks to keep
    use_encoder_filter: bool = False  # Toggle encoder-based filtering
    filter_only: bool = False  # Run only the filter step, skip LLM generation
    # Query augmentation options
    use_query_augmentation: bool = False  # Generate paraphrased queries for filtering
    num_augmented_queries: int = 3  # Number of paraphrased queries to generate
    num_keywords: int = 5  # Number of keywords to generate
    query_augmentation_model: str = None  # Model for query augmentation (None = use main model)


class HFBenchmarker:
    def __init__(self, config: BenchmarkConfig):
        self.config = config
        self.model = None
        self.tokenizer = None
        self.judge_model = None
        self.judge_tokenizer = None
        self.encoder_model = None
        self.augmentation_model = None
        self.augmentation_tokenizer = None
        self.haystack_text = ""
        self.augmented_queries = None  # Cache for augmented queries
        self.generated_keywords = None  # Cache for generated keywords
        
        # In filter-only mode, we need a tokenizer but not the full LLM
        if self.config.filter_only:
            self._load_tokenizer_only()
        else:
            self._load_model()
            self._load_judge_model()
        
        self._load_augmentation_model()
        self._load_encoder_model()
        self._load_haystack()
        
        # Generate augmented queries if enabled
        if self.config.use_query_augmentation:
            self._generate_augmented_queries()

    def _load_tokenizer_only(self):
        """Load only the tokenizer without the full model (for filter-only mode)."""
        print(f"Loading tokenizer: {self.config.model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_name, token=access_token)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

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

    def _load_augmentation_model(self):
        """Load model for query augmentation if specified and different from main model."""
        if not self.config.use_query_augmentation:
            self.augmentation_model = None
            self.augmentation_tokenizer = None
            return
        
        if self.config.query_augmentation_model and self.config.query_augmentation_model != self.config.model_name:
            print(f"Loading query augmentation model: {self.config.query_augmentation_model}")
            if self.config.quantization:
                print(f"  Using {self.config.quantization} quantization")
            
            dtype = getattr(torch, self.config.torch_dtype)
            self.augmentation_tokenizer = AutoTokenizer.from_pretrained(
                self.config.query_augmentation_model, 
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
            
            self.augmentation_model = AutoModelForCausalLM.from_pretrained(
                self.config.query_augmentation_model,
                **load_kwargs
            )
            
            if self.augmentation_tokenizer.pad_token is None:
                self.augmentation_tokenizer.pad_token = self.augmentation_tokenizer.eos_token
        else:
            # Use the main model for augmentation
            self.augmentation_model = self.model
            self.augmentation_tokenizer = self.tokenizer

    def _generate_augmented_queries(self):
        """Generate paraphrased/augmented queries and keywords for the retrieval question.
        
        Uses two separate LLM calls for better reliability:
        1. First call generates paraphrased queries
        2. Second call generates keywords
        """
        if not self.config.use_query_augmentation:
            self.augmented_queries = [self.config.retrieval_question]
            self.generated_keywords = []
            return
        
        if self.augmentation_model is None:
            print("Warning: No augmentation model available. Using original query only.")
            self.augmented_queries = [self.config.retrieval_question]
            self.generated_keywords = []
            return
        
        import re
        
        # --- First LLM call: Generate paraphrased queries ---
        print(f"\nGenerating {self.config.num_augmented_queries} augmented queries...")
        
        paraphrase_messages = [
            {
                "role": "system",
                "content": "You are a helpful assistant that generates alternative phrasings of questions for information retrieval. Output only the paraphrased questions, one per line, numbered."
            },
            {
                "role": "user",
                "content": f"""Generate {self.config.num_augmented_queries} different paraphrased versions of the following question. 
Each paraphrase should express the same information need but use different wording.

Original question: "{self.config.retrieval_question}"

Output format:
1. [first paraphrase]
2. [second paraphrase]
..."""
            }
        ]
        
        paraphrase_prompt = self._format_chat_prompt(paraphrase_messages, self.augmentation_tokenizer)
        inputs = self.augmentation_tokenizer(paraphrase_prompt, return_tensors="pt", truncation=True)
        inputs = inputs.to(self.augmentation_model.device)
        
        with torch.no_grad():
            outputs = self.augmentation_model.generate(
                **inputs,
                max_new_tokens=256,
                do_sample=False,
                # temperature=0.7,
                # top_p=0.9,
                temperature=None,
                top_p=None,
                pad_token_id=self.augmentation_tokenizer.pad_token_id
            )
        
        paraphrase_response = self.augmentation_tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1]:], 
            skip_special_tokens=True
        ).strip()
        
        # Parse paraphrases
        augmented = []
        for line in paraphrase_response.strip().split('\n'):
            line = line.strip()
            if not line:
                continue
            # Remove numbering patterns
            cleaned = re.sub(r'^[\d]+[.\):\-]\s*', '', line).strip()
            # Skip if it looks like a header or instruction
            if cleaned.upper().startswith(('PARAPHRAS', 'QUESTION', 'OUTPUT', 'ORIGINAL')):
                continue
            if cleaned and len(cleaned) > 10:
                augmented.append(cleaned)
        
        augmented = augmented[:self.config.num_augmented_queries]
        
        # --- Second LLM call: Generate keywords ---
        keywords = []
        if self.config.num_keywords > 0:
            print(f"Generating {self.config.num_keywords} keywords...")
            
            keyword_messages = [
                {
                    "role": "system",
                    "content": "You are a helpful assistant that extracts relevant keywords and short phrases for information retrieval. Output only the keywords, one per line, numbered."
                },
                {
                    "role": "user",
                    "content": f"""Extract {self.config.num_keywords} relevant keywords or short phrases that would help find documents answering this question.
Focus on key terms, names, concepts, and distinctive phrases.

Question: "{self.config.retrieval_question}"

Output format:
1. [first keyword or phrase]
2. [second keyword or phrase]
..."""
                }
            ]
            
            keyword_prompt = self._format_chat_prompt(keyword_messages, self.augmentation_tokenizer)
            inputs = self.augmentation_tokenizer(keyword_prompt, return_tensors="pt", truncation=True)
            inputs = inputs.to(self.augmentation_model.device)
            
            with torch.no_grad():
                outputs = self.augmentation_model.generate(
                    **inputs,
                    max_new_tokens=128,
                    do_sample=False,
                    # temperature=0.7,
                    # top_p=0.9,
                    temperature=None,
                    top_p=None,
                    pad_token_id=self.augmentation_tokenizer.pad_token_id
                )
            
            keyword_response = self.augmentation_tokenizer.decode(
                outputs[0][inputs["input_ids"].shape[1]:], 
                skip_special_tokens=True
            ).strip()
            
            # Parse keywords
            for line in keyword_response.strip().split('\n'):
                line = line.strip()
                if not line:
                    continue
                # Remove numbering patterns
                cleaned = re.sub(r'^[\d]+[.\):\-]\s*', '', line).strip()
                # Skip if it looks like a header or instruction
                if cleaned.upper().startswith(('KEYWORD', 'OUTPUT', 'QUESTION', 'PHRASE')):
                    continue
                if cleaned and len(cleaned) >= 2:
                    keywords.append(cleaned)
            
            keywords = keywords[:self.config.num_keywords]
        
        # Always include original query
        self.augmented_queries = [self.config.retrieval_question] + augmented
        self.generated_keywords = keywords
        
        print(f"  Original: {self.config.retrieval_question}")
        for i, q in enumerate(augmented, 1):
            print(f"  Paraphrase {i}: {q}")
        for i, kw in enumerate(keywords, 1):
            print(f"  Keyword {i}: {kw}")
        print(f"  Total queries for filtering: {len(self.augmented_queries)} + {len(self.generated_keywords)} keywords")

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

    def _format_chat_prompt(self, messages: list, tokenizer, add_generation_prompt: bool = True) -> str:
        """Format messages using the tokenizer's chat template.
        
        Args:
            messages: List of message dicts with 'role' and 'content' keys.
                     Roles can be 'system', 'user', 'assistant'.
            tokenizer: The tokenizer to use for formatting.
            add_generation_prompt: Whether to add the generation prompt at the end.
        
        Returns:
            Formatted prompt string.
        """
        # Use the tokenizer's chat template if available
        if hasattr(tokenizer, 'apply_chat_template') and tokenizer.chat_template:
            return tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=add_generation_prompt
            )
        
        # Fallback: Build a simple format that works for most models
        prompt_parts = []
        for msg in messages:
            role = msg["role"]
            content = msg["content"]
            if role == "system":
                prompt_parts.append(f"System: {content}\n")
            elif role == "user":
                prompt_parts.append(f"User: {content}\n")
            elif role == "assistant":
                prompt_parts.append(f"Assistant: {content}\n")
        
        if add_generation_prompt:
            prompt_parts.append("Assistant: ")
        
        return "".join(prompt_parts)

    def _split_into_chunks(self, text: str, chunk_size: int, stride: int = None) -> list:
        """Split text into chunks of approximately chunk_size tokens with optional overlap.
        
        Args:
            text: The text to split into chunks.
            chunk_size: Number of tokens per chunk.
            stride: Step size between chunk starts. If None, defaults to chunk_size (no overlap).
                   A stride smaller than chunk_size creates overlapping chunks.
        """
        if stride is None:
            stride = chunk_size
        
        tokens = self.tokenizer.encode(text, add_special_tokens=False)
        chunks = []
        
        for i in range(0, len(tokens), stride):
            chunk_tokens = tokens[i:i + chunk_size]
            if chunk_tokens:
                chunk_text = self.tokenizer.decode(chunk_tokens, skip_special_tokens=True)
                chunks.append({
                    "text": chunk_text,
                    "start_idx": i,
                    "end_idx": i + len(chunk_tokens)
                })
            # Stop if we've reached the end
            if i + chunk_size >= len(tokens):
                break
        
        return chunks

    def _filter_context_with_encoder(self, context: str, question: str) -> str:
        """Filter context to keep only the most relevant chunks based on encoder similarity.
        
        If query augmentation is enabled, uses multiple query variants and keywords, then aggregates scores.
        """
        if not self.encoder_model:
            return context
        
        chunks = self._split_into_chunks(
            context, 
            self.config.filter_chunk_size,
            self.config.filter_chunk_stride
        )
        
        if len(chunks) <= self.config.filter_top_k:
            return context  # No need to filter if fewer chunks than top_k
        
        chunk_texts = [c["text"] for c in chunks]
        chunk_embeddings = self.encoder_model.encode(chunk_texts, convert_to_tensor=True)
        chunk_embeddings = F.normalize(chunk_embeddings, p=2, dim=1)
        
        # Get queries and keywords to use for filtering
        if self.config.use_query_augmentation and self.augmented_queries:
            queries = self.augmented_queries.copy()
            # Add keywords as additional queries
            if self.generated_keywords:
                queries.extend(self.generated_keywords)
        else:
            queries = [question]
        
        # Encode all queries/keywords and compute similarities
        query_embeddings = self.encoder_model.encode(queries, convert_to_tensor=True)
        query_embeddings = F.normalize(query_embeddings, p=2, dim=1)
        
        # Compute similarity for each query
        # Shape: [num_queries, num_chunks]
        all_similarities = torch.matmul(query_embeddings, chunk_embeddings.T)
        
        # Aggregate similarities across queries (max pooling to capture any relevant match)
        # Alternative strategies: mean, weighted sum
        aggregated_similarities = all_similarities.mean(dim=0)
        
        # Get top-k chunk indices
        k = min(self.config.filter_top_k, len(chunks))
        topk_indices = torch.topk(aggregated_similarities, k=k).indices.tolist()
        
        # Sort indices to maintain original order
        topk_indices = sorted(topk_indices)
        
        # Reconstruct filtered context
        selected_chunks = [chunks[i]["text"] for i in topk_indices]
        filtered_context = "\n\n".join(selected_chunks)
        
        return filtered_context

    def _build_context(self, context_length: int, needle_depth: float) -> tuple:
        """Build context with needle inserted at specified depth. Returns (full_context, filtered_context, filter_latency)."""
        tokens = self.tokenizer.encode(self.haystack_text, add_special_tokens=False)
        target_tokens = context_length - len(self.tokenizer.encode(self.config.needle, add_special_tokens=False)) - 50
        tokens = tokens[:max(target_tokens, 100)]
        text = self.tokenizer.decode(tokens, skip_special_tokens=True)
        
        # Insert needle at depth
        insert_pos = int(len(text) * needle_depth)
        full_context = text[:insert_pos] + f"\n\n{self.config.needle}\n\n" + text[insert_pos:]
        
        # Apply encoder filtering if enabled
        filter_latency = 0.0
        if self.config.use_encoder_filter and self.encoder_model:
            filter_start = datetime.now()
            filtered_context = self._filter_context_with_encoder(
                full_context, 
                self.config.retrieval_question
            )
            filter_latency = (datetime.now() - filter_start).total_seconds()
        else:
            filtered_context = full_context
        
        return full_context, filtered_context, filter_latency

    def _build_prompt(self, context: str) -> str:
        """Build the full prompt with context and question using the model's chat template."""
        messages = [
            {
                "role": "system",
                "content": "You are a helpful assistant. Answer questions based only on the provided context. Be concise and accurate."
            },
            {
                "role": "user",
                "content": f"Context:\n{context}\n\nQuestion: {self.config.retrieval_question}\n\nAnswer based only on the context provided."
            }
        ]
        
        return self._format_chat_prompt(messages, self.tokenizer)

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
                temperature=None,
                top_p=None,
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
        messages = [
            {
                "role": "system",
                "content": "You are an expert evaluator assessing response accuracy. You always respond with a JSON object containing 'score' (integer 1-10) and 'reasoning' (brief explanation)."
            },
            {
                "role": "user",
                "content": f"""Evaluate this response for accuracy in retrieving specific information.

Target information (the "needle"): "{self.config.needle}"

Question asked: "{self.config.retrieval_question}"

Model's response: "{response}"

Scoring criteria:
- Score 1-2: Response is unrelated, wrong, or contradicts the needle
- Score 3-4: Response mentions the topic but misses key details
- Score 5-6: Response captures some elements but is incomplete or partially incorrect
- Score 7-8: Response is mostly accurate with minor omissions
- Score 9-10: Response accurately retrieves and presents the needle information

Respond with ONLY a JSON object: {{"score": <1-10>, "reasoning": "<brief explanation>"}}"""
            }
        ]
        
        judge_prompt = self._format_chat_prompt(messages, self.judge_tokenizer)

        inputs = self.judge_tokenizer(judge_prompt, return_tensors="pt", truncation=True).to(self.judge_model.device)
        with torch.no_grad():
            outputs = self.judge_model.generate(
                **inputs,
                max_new_tokens=150,
                do_sample=False,
                temperature=None,
                top_p=None,
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
                
                full_context, filtered_context, filter_latency = self._build_context(ctx_len, depth)
                prompt = self._build_prompt(filtered_context)
                
                # Calculate context stats
                full_tokens = len(self.tokenizer.encode(full_context, add_special_tokens=False))
                filtered_tokens = len(self.tokenizer.encode(filtered_context, add_special_tokens=False))
                
                if self.config.use_encoder_filter:
                    print(f"  Context: {full_tokens} -> {filtered_tokens} tokens (filtered in {filter_latency:.2f}s)")
                
                gen_start_time = datetime.now()
                response = self._generate(prompt)
                generation_latency = (datetime.now() - gen_start_time).total_seconds()
                
                total_latency = filter_latency + generation_latency
                
                eval_result = self._evaluate(response)
                
                result = {
                    "context_length": ctx_len,
                    "needle_depth": depth,
                    "response": response,
                    "filter_latency_seconds": filter_latency,
                    "generation_latency_seconds": generation_latency,
                    "total_latency_seconds": total_latency,
                    "full_context_tokens": full_tokens,
                    "filtered_context_tokens": filtered_tokens,
                    "encoder_filtered": self.config.use_encoder_filter,
                    **eval_result
                }
                results.append(result)
                
                if self.config.skip_evaluation:
                    print(f"  Evaluation: skipped | Gen: {generation_latency:.2f}s | Total: {total_latency:.2f}s")
                else:
                    score_display = eval_result.get('score', eval_result.get('score_normalized', 0) * 10)
                    print(f"  Score: {score_display}/10 | Gen: {generation_latency:.2f}s | Total: {total_latency:.2f}s")
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

    def run_filter_only(self) -> list:
        """Run only the filtering step without LLM generation. Saves filtered/full contexts to files.
        
        If no encoder model is provided, saves full (unfiltered) contexts instead.
        """
        # Determine if we're actually filtering or just saving full contexts
        use_filtering = self.config.use_encoder_filter and self.encoder_model is not None
        
        results = []
        os.makedirs(self.config.output_dir, exist_ok=True)
        
        # Create contexts subdirectory for text files
        contexts_dir = os.path.join(self.config.output_dir, "filtered_contexts" if use_filtering else "full_contexts")
        os.makedirs(contexts_dir, exist_ok=True)
        
        total_runs = len(self.config.context_lengths) * len(self.config.needle_depths)
        run_idx = 0
        
        print("\n=== Filter-Only Mode ===")
        if use_filtering:
            print(f"Encoder model: {self.config.encoder_model_name}")
            print(f"Chunk size: {self.config.filter_chunk_size}, Stride: {self.config.filter_chunk_stride or self.config.filter_chunk_size}, Top-k: {self.config.filter_top_k}")
            if self.config.use_query_augmentation and self.augmented_queries:
                num_kw = len(self.generated_keywords) if self.generated_keywords else 0
                print(f"Query augmentation: enabled ({len(self.augmented_queries)} queries + {num_kw} keywords)")
        else:
            print("No encoder model - saving full (unfiltered) contexts")
        
        for ctx_len in self.config.context_lengths:
            for depth in self.config.needle_depths:
                run_idx += 1
                print(f"\n[{run_idx}/{total_runs}] Context: {ctx_len} tokens, Depth: {depth:.0%}")
                
                full_context, filtered_context, filter_latency = self._build_context(ctx_len, depth)
                
                # When no encoder, use full context
                context_to_save = filtered_context if use_filtering else full_context
                
                # Calculate context stats
                full_tokens = len(self.tokenizer.encode(full_context, add_special_tokens=False))
                saved_tokens = len(self.tokenizer.encode(context_to_save, add_special_tokens=False))
                
                # Check if needle is preserved
                needle_preserved = self.config.needle in context_to_save
                
                if use_filtering:
                    print(f"  Context: {full_tokens} -> {saved_tokens} tokens ({filter_latency:.2f}s)")
                else:
                    print(f"  Context: {full_tokens} tokens (full, unfiltered)")
                print(f"  Needle preserved: {needle_preserved}")
                
                # Save context to file
                context_filename = f"ctx{ctx_len}_depth{int(depth*100)}.txt"
                context_path = os.path.join(contexts_dir, context_filename)
                with open(context_path, "w", encoding="utf-8") as f:
                    f.write(context_to_save)
                
                result = {
                    "context_length": ctx_len,
                    "needle_depth": depth,
                    "filter_latency_seconds": filter_latency if use_filtering else 0.0,
                    "full_context_tokens": full_tokens,
                    "saved_context_tokens": saved_tokens,
                    "compression_ratio": saved_tokens / full_tokens if full_tokens > 0 else 1.0,
                    "needle_preserved": needle_preserved,
                    "context_file": context_filename,
                    "filtered": use_filtering
                }
                results.append(result)
        
        # Save results summary
        encoder_name = self.config.encoder_model_name.replace('/', '_') if use_filtering else "no_filter"
        output_path = os.path.join(
            self.config.output_dir,
            f"filter_results_{encoder_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        )
        
        # Include augmented queries in output
        output_data = {
            "config": self.config.__dict__, 
            "results": results,
            "filtering_enabled": use_filtering
        }
        if self.config.use_query_augmentation and self.augmented_queries:
            output_data["augmented_queries"] = self.augmented_queries
            output_data["generated_keywords"] = self.generated_keywords if self.generated_keywords else []
        
        with open(output_path, "w") as f:
            json.dump(output_data, f, indent=2)
        
        # Print summary
        preserved_count = sum(1 for r in results if r["needle_preserved"])
        avg_compression = sum(r["compression_ratio"] for r in results) / len(results) if results else 0
        
        print(f"\n=== Summary ===")
        print(f"Needle preserved: {preserved_count}/{len(results)} ({100*preserved_count/len(results):.1f}%)")
        if use_filtering:
            avg_latency = sum(r["filter_latency_seconds"] for r in results) / len(results) if results else 0
            print(f"Avg compression ratio: {avg_compression:.2%}")
            print(f"Avg filter latency: {avg_latency:.3f}s")
        print(f"Results saved to: {output_path}")
        print(f"Contexts saved to: {contexts_dir}/")
        
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
    parser.add_argument("--filter-chunk-stride", type=int, default=None, help="Stride between chunks (default: same as chunk-size, no overlap). Use smaller value for overlapping chunks.")
    parser.add_argument("--filter-top-k", type=int, default=5, help="Number of top chunks to keep after filtering")
    parser.add_argument("--filter-only", action="store_true", help="Run only the filter step without LLM generation. Saves filtered contexts to files.")
    parser.add_argument("--use-query-augmentation", action="store_true", help="Generate paraphrased queries to improve filtering")
    parser.add_argument("--num-augmented-queries", type=int, default=3, help="Number of paraphrased queries to generate (default: 3)")
    parser.add_argument("--num-keywords", type=int, default=5, help="Number of keywords to generate for query augmentation (default: 5)")
    parser.add_argument("--query-augmentation-model", type=str, default=None, help="Model for query augmentation (default: use main model)")
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
        filter_chunk_stride=args.filter_chunk_stride,
        filter_top_k=args.filter_top_k,
        filter_only=args.filter_only,
        use_query_augmentation=args.use_query_augmentation,
        num_augmented_queries=args.num_augmented_queries,
        num_keywords=args.num_keywords,
        query_augmentation_model=args.query_augmentation_model
    )
    
    benchmarker = HFBenchmarker(config)
    
    if args.filter_only:
        benchmarker.run_filter_only()
    else:
        benchmarker.run()


if __name__ == "__main__":
    main()
