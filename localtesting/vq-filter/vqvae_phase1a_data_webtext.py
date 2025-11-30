"""
Phase 1a Data Loading: Web Text Only (Curriculum Learning)

This module loads only web/internet text for stable base VQ-VAE training.
After training converges, use Phase 1b to fine-tune on academic + conversational data.

Datasets:
- WikiText-103 (Wikipedia articles)
- C4 (Common Crawl web text)
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


def load_webtext(
    verbose: bool = True,
    split: str = "train",
    val_ratio: float = 0.05,
    seed: int = 42,
    c4_limit: int = 200_000,
) -> List[str]:
    """
    Load web text only (homogeneous domain for stable training).
    
    Args:
        verbose: Whether to print loading progress.
        split: "train" or "val" - which split to return.
        val_ratio: Fraction of data to use for validation.
        seed: Random seed for reproducible train/val split.
        c4_limit: Maximum number of C4 documents to load.
    
    Returns:
        List of text strings for the specified split.
    """
    all_texts = []
    
    if verbose:
        print("\n" + "="*60)
        print("LOADING WEB TEXT (Phase 1a - Curriculum Learning)")
        print("="*60)
    
    # Load wikitext-103-raw-v1
    try:
        if verbose:
            print("Loading wikitext-103-raw-v1...")
        wikitext = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1", split="train")
        wikitext_texts = [ex["text"] for ex in wikitext if ex["text"].strip()]
        all_texts.extend(wikitext_texts)
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
            all_texts.extend(wikitext_texts)
            if verbose:
                print(f"  Loaded {len(wikitext_texts)} texts from wikitext-2")
        except Exception as e2:
            if verbose:
                print(f"  Could not load wikitext-2 either: {e2}")
    
    # Load C4 (web text)
    try:
        if verbose:
            print(f"Loading c4 (streaming, ~{c4_limit//1000}k docs)...")
        c4 = load_dataset("allenai/c4", "en", split="train", streaming=True)
        c4_texts = []
        for i, ex in enumerate(c4):
            if ex["text"].strip():
                c4_texts.append(ex["text"])
            if i >= c4_limit:
                break
        all_texts.extend(c4_texts)
        if verbose:
            print(f"  Loaded {len(c4_texts)} texts from c4")
    except Exception as e:
        if verbose:
            print(f"  Could not load c4: {e}")
    
    if verbose:
        print(f"\nTotal web texts loaded: {len(all_texts)}")
    
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
    
    # Shuffle again with different seed per split
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
    batch_size: int = 256,
    num_workers: int = 4,
    tokenizer_name: str = "gpt2",
    window_size: int = 64,
    kmer_len: int = 4,
    stride: int = 32,
    max_tokens_total: int = 50_000_000,
    verbose: bool = True,
    split: str = "train",
    c4_limit: int = 200_000,
) -> DataLoader:
    """
    Create a DataLoader for web text only.
    """
    if verbose:
        print(f"Loading web texts for dataset ({split} split)...")
    texts = load_webtext(verbose=verbose, split=split, c4_limit=c4_limit)
    
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
    print("Testing Phase 1a (Web Text Only) dataloader...")
    dl = make_dataloader(batch_size=4, num_workers=0, max_tokens_total=10000, c4_limit=1000)
    
    for i, batch in enumerate(dl):
        print(f"Batch {i}:")
        print(f"  tokens shape: {batch['tokens'].shape}")
        print(f"  kmer_start: {batch['kmer_start']}")
        print(f"  kmer_len: {batch['kmer_len']}")
        if i >= 2:
            break
    
    print("\nPhase 1a data loading test complete!")
