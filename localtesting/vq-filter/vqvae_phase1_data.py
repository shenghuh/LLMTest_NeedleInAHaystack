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


def load_texts(verbose: bool = True) -> List[str]:
    """
    Load and combine texts from multiple datasets.
    Returns a shuffled list of text strings.
    """
    all_texts = []
    
    # Load wikitext-103-raw-v1 (larger than wikitext-2)
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
    
    # Load openwebtext (subset)
    try:
        if verbose:
            print("Loading openwebtext (10% subset)...")
        owt = load_dataset("Skylion007/openwebtext", split="train[:10%]")
        owt_texts = [ex["text"] for ex in owt if ex["text"].strip()]
        all_texts.extend(owt_texts)
        if verbose:
            print(f"  Loaded {len(owt_texts)} texts from openwebtext")
    except Exception as e:
        if verbose:
            print(f"  Could not load openwebtext: {e}")
    
    # Load c4 (small subset for diversity)
    try:
        if verbose:
            print("Loading c4 (1% subset)...")
        c4 = load_dataset("allenai/c4", "en", split="train", streaming=True)
        c4_texts = []
        for i, ex in enumerate(c4):
            if ex["text"].strip():
                c4_texts.append(ex["text"])
            if i >= 50000:  # Limit to ~50k documents
                break
        all_texts.extend(c4_texts)
        if verbose:
            print(f"  Loaded {len(c4_texts)} texts from c4")
    except Exception as e:
        if verbose:
            print(f"  Could not load c4: {e}")
    
    if verbose:
        print(f"Total texts loaded: {len(all_texts)}")
    
    # Shuffle all texts
    random.shuffle(all_texts)
    
    return all_texts


class KmerWindowDataset(IterableDataset):
    """
    Iterable dataset that yields fixed-size token windows with k-mer positions.
    """
    
    def __init__(
        self,
        texts: List[str],
        tokenizer_name: str = "gpt2",
        window_size: int = 256,
        kmer_len: int = 16,
        stride: int = 128,
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
    window_size: int = 256,
    kmer_len: int = 16,
    stride: int = 128,
    max_tokens_total: int = 50_000_000,
    verbose: bool = True,
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
    
    Returns:
        DataLoader yielding batches of token windows with k-mer positions.
    """
    if verbose:
        print("Loading texts for dataset...")
    texts = load_texts(verbose=verbose)
    
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
