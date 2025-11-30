"""
Phase 1b Data Loading: Academic Papers + Conversational Text

This module loads academic and conversational text for fine-tuning
a VQ-VAE that was pre-trained on web text (Phase 1a).

Use this AFTER Phase 1a training has converged on web text.

Datasets:
- Academic: arXiv, PubMed
- Conversational: OpenAssistant, UltraChat, Alpaca, HH-RLHF, ShareGPT
"""

import random
import warnings
from typing import List

import torch
from torch import LongTensor
from torch.utils.data import DataLoader, IterableDataset
from datasets import load_dataset
from transformers import AutoTokenizer

# Suppress dataset loading warnings
warnings.filterwarnings("ignore", message=".*trust_remote_code.*")
warnings.filterwarnings("ignore", message=".*loading script.*")


def load_academic_texts(verbose: bool = True) -> List[str]:
    """Load academic papers (arXiv + PubMed)."""
    paper_texts = []
    
    if verbose:
        print("\n" + "="*60)
        print("LOADING ACADEMIC PAPERS")
        print("="*60)
    
    # Load arXiv
    try:
        if verbose:
            print("Loading arXiv (ccdv/arxiv-summarization)...")
        arxiv = load_dataset("ccdv/arxiv-summarization", split="train")
        arxiv_texts = []
        for ex in arxiv:
            if "article" in ex and ex["article"].strip():
                arxiv_texts.append(ex["article"])
            elif "abstract" in ex and ex["abstract"].strip():
                arxiv_texts.append(ex["abstract"])
        paper_texts.extend(arxiv_texts)
        if verbose:
            print(f"  Loaded {len(arxiv_texts)} texts from arXiv")
    except Exception as e:
        if verbose:
            print(f"  Could not load arXiv: {e}")
    
    # Load PubMed
    try:
        if verbose:
            print("Loading PubMed (ccdv/pubmed-summarization)...")
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
    
    if verbose:
        print(f"  Total academic texts: {len(paper_texts)}")
    
    return paper_texts


def load_conversational_texts(verbose: bool = True) -> List[str]:
    """Load conversational text from multiple sources."""
    conv_texts = []
    
    if verbose:
        print("\n" + "="*60)
        print("LOADING CONVERSATIONAL TEXT")
        print("="*60)
    
    # OpenAssistant
    try:
        if verbose:
            print("Loading OpenAssistant (oasst1)...")
        oasst = load_dataset("OpenAssistant/oasst1", split="train")
        oasst_texts = [ex["text"] for ex in oasst if ex.get("text", "").strip()]
        conv_texts.extend(oasst_texts)
        if verbose:
            print(f"  Loaded {len(oasst_texts)} texts from OpenAssistant")
    except Exception as e:
        if verbose:
            print(f"  Could not load OpenAssistant: {e}")
    
    # UltraChat (streaming)
    try:
        if verbose:
            print("Loading UltraChat (streaming, ~50k convos)...")
        ultrachat = load_dataset("stingning/ultrachat", split="train", streaming=True)
        ultrachat_texts = []
        for i, ex in enumerate(ultrachat):
            if "data" in ex and ex["data"]:
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
    
    # Alpaca
    try:
        if verbose:
            print("Loading Alpaca (tatsu-lab/alpaca)...")
        alpaca = load_dataset("tatsu-lab/alpaca", split="train")
        alpaca_texts = []
        for ex in alpaca:
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
    
    # Anthropic HH-RLHF
    try:
        if verbose:
            print("Loading Anthropic HH-RLHF...")
        hh = load_dataset("Anthropic/hh-rlhf", split="train")
        hh_texts = [ex["chosen"] for ex in hh if ex.get("chosen", "").strip()]
        conv_texts.extend(hh_texts)
        if verbose:
            print(f"  Loaded {len(hh_texts)} texts from HH-RLHF")
    except Exception as e:
        if verbose:
            print(f"  Could not load HH-RLHF: {e}")
    
    # ShareGPT Vicuna
    try:
        if verbose:
            print("Loading ShareGPT Vicuna (filtered)...")
        sharegpt = load_dataset(
            "anon8231489123/ShareGPT_Vicuna_unfiltered",
            data_files="ShareGPT_V3_unfiltered_cleaned_split.json",
            split="train"
        )
        sharegpt_texts = []
        for ex in sharegpt:
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
    
    if verbose:
        print(f"  Total conversational texts: {len(conv_texts)}")
    
    return conv_texts


def load_domain_texts(
    domain: str = "both",
    verbose: bool = True,
    split: str = "train",
    val_ratio: float = 0.05,
    seed: int = 42,
) -> List[str]:
    """
    Load academic and/or conversational texts.
    
    Args:
        domain: "academic", "conversational", or "both"
        verbose: Whether to print loading progress.
        split: "train" or "val" - which split to return.
        val_ratio: Fraction of data to use for validation.
        seed: Random seed for reproducible train/val split.
    
    Returns:
        List of text strings for the specified split.
    """
    all_texts = []
    
    if domain in ["academic", "both"]:
        all_texts.extend(load_academic_texts(verbose=verbose))
    
    if domain in ["conversational", "both"]:
        all_texts.extend(load_conversational_texts(verbose=verbose))
    
    if verbose:
        print(f"\nTotal texts loaded: {len(all_texts)}")
    
    # Deterministic shuffle
    rng = random.Random(seed)
    rng.shuffle(all_texts)
    
    # Split into train and val
    n_val = int(len(all_texts) * val_ratio)
    if split == "val":
        texts = all_texts[:n_val]
        if verbose:
            print(f"Using validation split: {len(texts)} texts")
    else:
        texts = all_texts[n_val:]
        if verbose:
            print(f"Using training split: {len(texts)} texts")
    
    # Shuffle again
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
        super().__init__()
        self.texts = texts
        self.tokenizer_name = tokenizer_name
        self.window_size = window_size
        self.kmer_len = kmer_len
        self.stride = stride
        self.max_tokens_total = max_tokens_total
        
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
    
    def __iter__(self):
        tokens_processed = 0
        
        for text in self.texts:
            if tokens_processed >= self.max_tokens_total:
                break
            
            encoded = self.tokenizer.encode(
                text,
                add_special_tokens=False,
                truncation=False,
            )
            
            if len(encoded) < self.window_size:
                continue
            
            for start in range(0, len(encoded) - self.window_size + 1, self.stride):
                if tokens_processed >= self.max_tokens_total:
                    break
                
                window = encoded[start : start + self.window_size]
                
                margin = self.window_size // 4
                min_kmer_start = margin
                max_kmer_start = self.window_size - margin - self.kmer_len
                
                if max_kmer_start <= min_kmer_start:
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
    domain: str = "both",
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
    Create a DataLoader for academic/conversational text.
    
    Args:
        domain: "academic", "conversational", or "both"
    """
    if verbose:
        print(f"Loading {domain} texts ({split} split)...")
    texts = load_domain_texts(domain=domain, verbose=verbose, split=split)
    
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
    print("Testing Phase 1b (Academic + Conversational) dataloader...")
    
    # Test academic only
    print("\n--- Testing Academic Domain ---")
    dl = make_dataloader(domain="academic", batch_size=4, num_workers=0, max_tokens_total=10000)
    for i, batch in enumerate(dl):
        print(f"Batch {i}: tokens shape {batch['tokens'].shape}")
        if i >= 1:
            break
    
    # Test conversational only
    print("\n--- Testing Conversational Domain ---")
    dl = make_dataloader(domain="conversational", batch_size=4, num_workers=0, max_tokens_total=10000)
    for i, batch in enumerate(dl):
        print(f"Batch {i}: tokens shape {batch['tokens'].shape}")
        if i >= 1:
            break
    
    print("\nPhase 1b data loading test complete!")
