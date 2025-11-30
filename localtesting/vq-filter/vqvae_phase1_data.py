import random
import warnings
from typing import List, Optional

import torch
from torch import LongTensor
from torch.utils.data import DataLoader, IterableDataset
from datasets import load_dataset
from transformers import AutoTokenizer

# Suppress dataset loading warnings
warnings.filterwarnings("ignore", message=".*trust_remote_code.*")
warnings.filterwarnings("ignore", message=".*loading script.*")


def load_texts(verbose: bool = True, split: str = "train", val_ratio: float = 0.05, seed: int = 42) -> List[str]:
    """
    Load and combine texts from multiple datasets covering:
    - Internet text (wikitext, openwebtext, c4)
    - Academic papers (arxiv, pubmed)
    - Conversational text (OpenAssistant, ShareGPT-style, DailyDialog)
    
    Returns a shuffled list of text strings for the specified split.
    
    Args:
        verbose: Whether to print loading progress.
        split: "train" or "val" - which split to return.
        val_ratio: Fraction of data to use for validation.
        seed: Random seed for reproducible train/val split.
    
    Returns:
        List of text strings for the specified split.
    """
    all_texts = []
    domain_counts = {}
    
    # ==========================================================================
    # DOMAIN 1: INTERNET TEXT (~40% of data)
    # ==========================================================================
    if verbose:
        print("\n" + "="*60)
        print("LOADING INTERNET TEXT DOMAIN")
        print("="*60)
    
    internet_texts = []
    
    # Load wikitext-103-raw-v1 (larger than wikitext-2)
    try:
        if verbose:
            print("Loading wikitext-103-raw-v1...")
        wikitext = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1", split="train")
        wikitext_texts = [ex["text"] for ex in wikitext if ex["text"].strip()]
        internet_texts.extend(wikitext_texts)
        if verbose:
            print(f"  Loaded {len(wikitext_texts)} texts from wikitext-103")
    except Exception as e:
        if verbose:
            print(f"  Could not load wikitext-103: {e}")
        # Fallback to wikitext-2
        try:
            if verbose:
                print("  Trying wikitext-2-raw-v1 as fallback...")
            wikitext = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="train")
            wikitext_texts = [ex["text"] for ex in wikitext if ex["text"].strip()]
            internet_texts.extend(wikitext_texts)
            if verbose:
                print(f"  Loaded {len(wikitext_texts)} texts from wikitext-2")
        except Exception as e2:
            if verbose:
                print(f"  Could not load wikitext-2 either: {e2}")
    
    # Load c4 (web text, replaces openwebtext)
    try:
        if verbose:
            print("Loading c4 (streaming, ~200k docs)...")
        c4 = load_dataset("allenai/c4", "en", split="train", streaming=True)
        c4_texts = []
        for i, ex in enumerate(c4):
            if ex["text"].strip():
                c4_texts.append(ex["text"])
            if i >= 200000:  # Increased from 50k to compensate for removed openwebtext
                break
        internet_texts.extend(c4_texts)
        if verbose:
            print(f"  Loaded {len(c4_texts)} texts from c4")
    except Exception as e:
        if verbose:
            print(f"  Could not load c4: {e}")
    
    domain_counts["internet"] = len(internet_texts)
    all_texts.extend(internet_texts)
    
    # ==========================================================================
    # DOMAIN 2: ACADEMIC PAPERS (~30% of data)
    # ==========================================================================
    if verbose:
        print("\n" + "="*60)
        print("LOADING ACADEMIC PAPERS DOMAIN")
        print("="*60)
    
    paper_texts = []
    
    # Load arXiv abstracts/summaries
    try:
        if verbose:
            print("Loading arXiv abstracts (ccdv/arxiv-summarization)...")
        arxiv = load_dataset("ccdv/arxiv-summarization", split="train")
        arxiv_texts = []
        for ex in arxiv:
            # Use article text (full paper) for rich context
            if "article" in ex and ex["article"].strip():
                arxiv_texts.append(ex["article"])
            # Also include abstract for shorter samples
            elif "abstract" in ex and ex["abstract"].strip():
                arxiv_texts.append(ex["abstract"])
        paper_texts.extend(arxiv_texts)
        if verbose:
            print(f"  Loaded {len(arxiv_texts)} texts from arXiv")
    except Exception as e:
        if verbose:
            print(f"  Could not load arXiv: {e}")
    
    # Load PubMed abstracts
    try:
        if verbose:
            print("Loading PubMed abstracts (ccdv/pubmed-summarization)...")
        pubmed = load_dataset("ccdv/pubmed-summarization", split="train")
        pubmed_texts = []
        for ex in pubmed:
            if "article" in ex and ex["article"].strip():
                pubmed_texts.append(ex["article"])
            elif "abstract" in ex and ex["abstract"].strip():
                pubmed_texts.append(ex["abstract"])
        paper_texts.extend(pubmed_texts)
        if verbose:
            print(f"  Loaded {len(pubmed_texts)} texts from PubMed")
    except Exception as e:
        if verbose:
            print(f"  Could not load PubMed: {e}")
    
    # Alternative: scientific_papers if above fail
    if len(paper_texts) == 0:
        try:
            if verbose:
                print("Trying scientific_papers as fallback...")
            sci = load_dataset("scientific_papers", "arxiv", split="train[:20%]", trust_remote_code=True)
            sci_texts = [ex["article"] for ex in sci if ex["article"].strip()]
            paper_texts.extend(sci_texts)
            if verbose:
                print(f"  Loaded {len(sci_texts)} texts from scientific_papers")
        except Exception as e:
            if verbose:
                print(f"  Could not load scientific_papers: {e}")
    
    domain_counts["papers"] = len(paper_texts)
    all_texts.extend(paper_texts)
    
    # ==========================================================================
    # DOMAIN 3: CONVERSATIONAL TEXT (~30% of data)
    # ==========================================================================
    if verbose:
        print("\n" + "="*60)
        print("LOADING CONVERSATIONAL TEXT DOMAIN")
        print("="*60)
    
    conv_texts = []
    
    # Load OpenAssistant conversations
    try:
        if verbose:
            print("Loading OpenAssistant (oasst1)...")
        oasst = load_dataset("OpenAssistant/oasst1", split="train")
        oasst_texts = []
        for ex in oasst:
            if "text" in ex and ex["text"].strip():
                oasst_texts.append(ex["text"])
        conv_texts.extend(oasst_texts)
        if verbose:
            print(f"  Loaded {len(oasst_texts)} texts from OpenAssistant")
    except Exception as e:
        if verbose:
            print(f"  Could not load OpenAssistant: {e}")
    
    # Load UltraChat (diverse multi-turn conversations)
    try:
        if verbose:
            print("Loading UltraChat (streaming, ~50k convos)...")
        ultrachat = load_dataset("stingning/ultrachat", split="train", streaming=True)
        ultrachat_texts = []
        for i, ex in enumerate(ultrachat):
            # UltraChat has "data" field with list of turns
            if "data" in ex and ex["data"]:
                # Concatenate turns into single text
                convo = "\n".join(ex["data"]) if isinstance(ex["data"], list) else str(ex["data"])
                if convo.strip():
                    ultrachat_texts.append(convo)
            if i >= 50000:
                break
        conv_texts.extend(ultrachat_texts)
        if verbose:
            print(f"  Loaded {len(ultrachat_texts)} texts from UltraChat")
    except Exception as e:
        if verbose:
            print(f"  Could not load UltraChat: {e}")
    
    # Load Alpaca (instruction-following conversations) - replaces DailyDialog
    try:
        if verbose:
            print("Loading Alpaca (tatsu-lab/alpaca)...")
        alpaca = load_dataset("tatsu-lab/alpaca", split="train")
        alpaca_texts = []
        for ex in alpaca:
            # Combine instruction, input, and output into conversation format
            parts = []
            instruction = ex.get("instruction", "")
            inp = ex.get("input", "")
            output = ex.get("output", "")
            
            if instruction.strip():
                if inp.strip():
                    parts.append(f"User: {instruction}\n{inp}")
                else:
                    parts.append(f"User: {instruction}")
            if output.strip():
                parts.append(f"Assistant: {output}")
            
            if parts:
                alpaca_texts.append("\n".join(parts))
        conv_texts.extend(alpaca_texts)
        if verbose:
            print(f"  Loaded {len(alpaca_texts)} texts from Alpaca")
    except Exception as e:
        if verbose:
            print(f"  Could not load Alpaca: {e}")
    
    # Load Anthropic HH-RLHF (use default config, not helpful-base)
    try:
        if verbose:
            print("Loading Anthropic HH-RLHF...")
        hh = load_dataset("Anthropic/hh-rlhf", split="train")
        hh_texts = []
        for ex in hh:
            if "chosen" in ex and ex["chosen"].strip():
                hh_texts.append(ex["chosen"])
        conv_texts.extend(hh_texts)
        if verbose:
            print(f"  Loaded {len(hh_texts)} texts from HH-RLHF")
    except Exception as e:
        if verbose:
            print(f"  Could not load HH-RLHF: {e}")
    
    # Load ShareGPT/Vicuna conversations (filtered version)
    try:
        if verbose:
            print("Loading ShareGPT Vicuna (filtered, ~90k convos)...")
        # Use the cleaned/filtered JSON file directly
        sharegpt = load_dataset(
            "anon8231489123/ShareGPT_Vicuna_unfiltered",
            data_files="ShareGPT_V3_unfiltered_cleaned_split.json",
            split="train"
        )
        sharegpt_texts = []
        for ex in sharegpt:
            # ShareGPT has "conversations" field with list of turns
            convos = ex.get("conversations", [])
            if convos:
                parts = []
                for turn in convos:
                    role = turn.get("from", "")
                    text = turn.get("value", "")
                    if text.strip():
                        parts.append(f"{role}: {text}")
                if parts:
                    sharegpt_texts.append("\n".join(parts))
        conv_texts.extend(sharegpt_texts)
        if verbose:
            print(f"  Loaded {len(sharegpt_texts)} texts from ShareGPT")
    except Exception as e:
        if verbose:
            print(f"  Could not load ShareGPT: {e}")
    
    # Load LMSYS Chat (diverse LLM conversations)
    try:
        if verbose:
            print("Loading LMSYS-Chat-1M (streaming, ~50k convos)...")
        lmsys = load_dataset("lmsys/lmsys-chat-1m", split="train", streaming=True)
        lmsys_texts = []
        for i, ex in enumerate(lmsys):
            # LMSYS has "conversation" field
            convo = ex.get("conversation", [])
            if convo:
                parts = []
                for turn in convo:
                    role = turn.get("role", "")
                    content = turn.get("content", "")
                    if content.strip():
                        parts.append(f"{role}: {content}")
                if parts:
                    lmsys_texts.append("\n".join(parts))
            if i >= 50000:
                break
        conv_texts.extend(lmsys_texts)
        if verbose:
            print(f"  Loaded {len(lmsys_texts)} texts from LMSYS-Chat")
    except Exception as e:
        if verbose:
            print(f"  Could not load LMSYS-Chat: {e}")
    
    domain_counts["conversational"] = len(conv_texts)
    all_texts.extend(conv_texts)
    
    # ==========================================================================
    # SUMMARY AND SPLIT
    # ==========================================================================
    if verbose:
        print("\n" + "="*60)
        print("DATASET SUMMARY")
        print("="*60)
        total = sum(domain_counts.values())
        for domain, count in domain_counts.items():
            pct = 100.0 * count / total if total > 0 else 0
            print(f"  {domain:15s}: {count:8d} ({pct:5.1f}%)")
        print(f"  {'TOTAL':15s}: {total:8d}")
    
    if verbose:
        print(f"\nTotal texts loaded: {len(all_texts)}")
    
    # Deterministic shuffle with seed for reproducible split
    rng = random.Random(seed)
    rng.shuffle(all_texts)
    
    # Split into train and val
    n_val = int(len(all_texts) * val_ratio)
    if split == "val":
        texts = all_texts[:n_val]
        if verbose:
            print(f"Using validation split: {len(texts)} texts")
    else:  # train
        texts = all_texts[n_val:]
        if verbose:
            print(f"Using training split: {len(texts)} texts")
    
    # Shuffle again with different seed per split for variety during training
    # (but still deterministic for reproducibility)
    split_seed = seed + (1 if split == "val" else 2)
    rng2 = random.Random(split_seed)
    rng2.shuffle(texts)
    
    return texts


class KmerWindowDataset(IterableDataset):
    """
    Iterable dataset that yields fixed-size token windows with k-mer positions.
    """
    
    def __init__(
        self,
        texts: List[str],
        tokenizer_name: str = "gpt2",
        window_size: int = 64,
        kmer_len: int = 4,
        stride: int = 32,
        max_tokens_total: int = 50_000_000,
    ):
        """
        Args:
            texts: List of text strings to process.
            tokenizer_name: HuggingFace tokenizer name.
            window_size: Size of each token window.
            kmer_len: Length of k-mer to reconstruct.
            stride: Stride for sliding window.
            max_tokens_total: Approximate budget for total tokens to process.
        """
        super().__init__()
        self.texts = texts
        self.tokenizer_name = tokenizer_name
        self.window_size = window_size
        self.kmer_len = kmer_len
        self.stride = stride
        self.max_tokens_total = max_tokens_total
        
        # Initialize tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
    
    def __iter__(self):
        tokens_processed = 0
        
        for text in self.texts:
            if tokens_processed >= self.max_tokens_total:
                break
            
            # Tokenize without special tokens
            encoded = self.tokenizer.encode(
                text,
                add_special_tokens=False,
                truncation=False,
            )
            
            if len(encoded) < self.window_size:
                continue
            
            # Slide window over token IDs
            for start in range(0, len(encoded) - self.window_size + 1, self.stride):
                if tokens_processed >= self.max_tokens_total:
                    break
                
                window = encoded[start : start + self.window_size]
                
                # Pick a central k-mer start index
                # Ensure k-mer fits fully and is not at extreme edges
                # Leave at least 1/4 of window on each side as margin
                margin = self.window_size // 4
                min_kmer_start = margin
                max_kmer_start = self.window_size - margin - self.kmer_len
                
                if max_kmer_start <= min_kmer_start:
                    # Fallback to centered position if window is too small
                    kmer_start = (self.window_size - self.kmer_len) // 2
                else:
                    kmer_start = random.randint(min_kmer_start, max_kmer_start)
                
                yield {
                    "tokens": LongTensor(window),
                    "kmer_start": LongTensor([kmer_start]).squeeze(),
                    "kmer_len": LongTensor([self.kmer_len]).squeeze(),
                }
                
                tokens_processed += self.window_size


def make_dataloader(
    batch_size: int = 256,
    num_workers: int = 4,
    tokenizer_name: str = "gpt2",
    window_size: int = 64,
    kmer_len: int = 4,
    stride: int = 32,
    max_tokens_total: int = 50_000_000,
    verbose: bool = True,
    split: str = "train",
) -> DataLoader:
    """
    Create a DataLoader from loaded texts.
    
    Args:
        batch_size: Batch size for the dataloader.
        num_workers: Number of worker processes.
        tokenizer_name: HuggingFace tokenizer name.
        window_size: Size of each token window.
        kmer_len: Length of k-mer to reconstruct.
        stride: Stride for sliding window.
        max_tokens_total: Approximate budget for total tokens.
        verbose: Whether to print loading progress.
        split: "train" or "val" - which data split to use.
    
    Returns:
        DataLoader yielding batches of token windows with k-mer positions.
    """
    if verbose:
        print(f"Loading texts for dataset ({split} split)...")
    texts = load_texts(verbose=verbose, split=split)
    
    if verbose:
        print("Creating KmerWindowDataset...")
    dataset = KmerWindowDataset(
        texts=texts,
        tokenizer_name=tokenizer_name,
        window_size=window_size,
        kmer_len=kmer_len,
        stride=stride,
        max_tokens_total=max_tokens_total,
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
    )
    
    return dataloader


if __name__ == "__main__":
    # Test the dataloader
    dl = make_dataloader(batch_size=4, num_workers=0, max_tokens_total=10000)
    
    for i, batch in enumerate(dl):
        print(f"Batch {i}:")
        print(f"  tokens shape: {batch['tokens'].shape}")
        print(f"  kmer_start: {batch['kmer_start']}")
        print(f"  kmer_len: {batch['kmer_len']}")
        if i >= 2:
            break
