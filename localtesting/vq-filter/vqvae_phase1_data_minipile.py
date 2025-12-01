"""
Phase 1 Data Loading: MiniPile Dataset

MiniPile is a 1M-example curated subset of The Pile, containing diverse text from:
- Academic: ArXiv, PubMed Abstracts, PubMed Central, PhilPapers, NIH ExPorter
- Code: GitHub, StackExchange
- Web: OpenWebText2, Pile-CC, Wikipedia
- Books: Books3, BookCorpus2, Gutenberg
- Conversational: HackerNews, Ubuntu IRC, EuroParl
- Other: USPTO, FreeLaw, DM Mathematics, Enron Emails

This provides diverse coverage in a single dataset, potentially eliminating 
the need for separate Phase 1a/1b curriculum learning.

Dataset: https://huggingface.co/datasets/JeanKaddour/minipile
"""

import random
import warnings
from typing import List, Dict, Iterator, Optional
from dataclasses import dataclass

import torch
from torch import LongTensor
from torch.utils.data import DataLoader, IterableDataset
from datasets import load_dataset
from transformers import AutoTokenizer

# Suppress warnings
warnings.filterwarnings("ignore", message=".*trust_remote_code.*")


@dataclass 
class MiniPileConfig:
    """Configuration for MiniPile data loading."""
    tokenizer_name: str = "gpt2"
    window_size: int = 64
    kmer_len: int = 4
    max_texts: int = 1_000_000  # Use all 1M examples
    min_text_length: int = 100  # Minimum characters
    shuffle_buffer: int = 10_000
    seed: int = 42


def load_minipile(
    split: str = "train",
    max_texts: Optional[int] = None,
    verbose: bool = True,
) -> List[str]:
    """
    Load MiniPile dataset.
    
    Args:
        split: Which split to load ("train", "validation", "test")
        max_texts: Maximum number of texts to load (None = all)
        verbose: Print loading progress
    
    Returns:
        List of text strings
    """
    if verbose:
        print(f"Loading MiniPile ({split} split)...")
    
    # Load dataset
    dataset = load_dataset(
        "JeanKaddour/minipile",
        split=split,
        trust_remote_code=True,
    )
    
    texts = []
    for i, example in enumerate(dataset):
        text = example.get("text", "")
        if text and len(text.strip()) >= 100:
            texts.append(text.strip())
        
        if max_texts and len(texts) >= max_texts:
            break
        
        if verbose and (i + 1) % 100_000 == 0:
            print(f"  Processed {i + 1:,} examples, kept {len(texts):,} texts")
    
    if verbose:
        print(f"  Loaded {len(texts):,} texts from MiniPile")
    
    return texts


class MiniPileDataset(IterableDataset):
    """
    Iterable dataset for MiniPile that yields tokenized windows.
    
    Each iteration yields a random window from the corpus with a 
    randomly selected k-mer position for reconstruction.
    """
    
    def __init__(
        self,
        texts: List[str],
        tokenizer_name: str = "gpt2",
        window_size: int = 64,
        kmer_len: int = 4,
        shuffle: bool = True,
        seed: int = 42,
    ):
        """
        Args:
            texts: List of text strings from MiniPile.
            tokenizer_name: HuggingFace tokenizer to use.
            window_size: Number of tokens per training window.
            kmer_len: Length of k-mer to reconstruct.
            shuffle: Whether to shuffle texts.
            seed: Random seed.
        """
        super().__init__()
        self.texts = texts
        self.tokenizer_name = tokenizer_name
        self.window_size = window_size
        self.kmer_len = kmer_len
        self.shuffle = shuffle
        self.seed = seed
        
        # Initialize tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.pad_token_id = self.tokenizer.pad_token_id
    
    def _process_text(self, text: str) -> Iterator[Dict[str, torch.Tensor]]:
        """Tokenize text and yield windows."""
        # Tokenize
        encoded = self.tokenizer.encode(
            text,
            add_special_tokens=False,
            truncation=False,
        )
        
        if len(encoded) < self.window_size // 2:
            return  # Skip very short texts
        
        # Pad if needed
        if len(encoded) < self.window_size:
            pad_len = self.window_size - len(encoded)
            encoded = encoded + [self.pad_token_id] * pad_len
        
        # Yield non-overlapping windows (with some randomness)
        num_windows = len(encoded) // self.window_size
        
        for w in range(num_windows):
            # Add small random offset for variety
            offset = random.randint(0, min(16, self.window_size // 4))
            start = w * self.window_size + offset
            
            if start + self.window_size > len(encoded):
                start = len(encoded) - self.window_size
            
            window = encoded[start : start + self.window_size]
            
            # Random k-mer start position (avoid edges)
            margin = self.window_size // 4
            min_kmer = margin
            max_kmer = self.window_size - margin - self.kmer_len
            
            if max_kmer <= min_kmer:
                kmer_start = (self.window_size - self.kmer_len) // 2
            else:
                kmer_start = random.randint(min_kmer, max_kmer)
            
            yield {
                "tokens": LongTensor(window),
                "kmer_start": LongTensor([kmer_start]).squeeze(),
                "kmer_len": LongTensor([self.kmer_len]).squeeze(),
            }
    
    def __iter__(self) -> Iterator[Dict[str, torch.Tensor]]:
        texts = self.texts.copy()
        
        if self.shuffle:
            rng = random.Random(self.seed)
            rng.shuffle(texts)
        
        for text in texts:
            yield from self._process_text(text)


class StreamingMiniPileDataset(IterableDataset):
    """
    Streaming version that loads MiniPile on-the-fly.
    More memory efficient for full dataset.
    """
    
    def __init__(
        self,
        tokenizer_name: str = "gpt2",
        window_size: int = 64,
        kmer_len: int = 4,
        split: str = "train",
        max_texts: Optional[int] = None,
        seed: int = 42,
    ):
        super().__init__()
        self.tokenizer_name = tokenizer_name
        self.window_size = window_size
        self.kmer_len = kmer_len
        self.split = split
        self.max_texts = max_texts
        self.seed = seed
        
        # Initialize tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.pad_token_id = self.tokenizer.pad_token_id
    
    def _process_text(self, text: str) -> Iterator[Dict[str, torch.Tensor]]:
        """Tokenize and yield windows from a single text."""
        encoded = self.tokenizer.encode(
            text,
            add_special_tokens=False,
            truncation=False,
        )
        
        if len(encoded) < self.window_size // 2:
            return
        
        if len(encoded) < self.window_size:
            pad_len = self.window_size - len(encoded)
            encoded = encoded + [self.pad_token_id] * pad_len
        
        num_windows = len(encoded) // self.window_size
        
        for w in range(num_windows):
            offset = random.randint(0, min(16, self.window_size // 4))
            start = w * self.window_size + offset
            
            if start + self.window_size > len(encoded):
                start = len(encoded) - self.window_size
            
            window = encoded[start : start + self.window_size]
            
            margin = self.window_size // 4
            min_kmer = margin
            max_kmer = self.window_size - margin - self.kmer_len
            
            if max_kmer <= min_kmer:
                kmer_start = (self.window_size - self.kmer_len) // 2
            else:
                kmer_start = random.randint(min_kmer, max_kmer)
            
            yield {
                "tokens": LongTensor(window),
                "kmer_start": LongTensor([kmer_start]).squeeze(),
                "kmer_len": LongTensor([self.kmer_len]).squeeze(),
            }
    
    def __iter__(self) -> Iterator[Dict[str, torch.Tensor]]:
        # Load dataset in streaming mode
        dataset = load_dataset(
            "JeanKaddour/minipile",
            split=self.split,
            streaming=True,
            trust_remote_code=True,
        )
        
        # Shuffle with buffer
        dataset = dataset.shuffle(seed=self.seed, buffer_size=10_000)
        
        count = 0
        for example in dataset:
            text = example.get("text", "")
            if not text or len(text.strip()) < 100:
                continue
            
            yield from self._process_text(text.strip())
            
            count += 1
            if self.max_texts and count >= self.max_texts:
                break


def make_minipile_dataloader(
    batch_size: int = 256,
    num_workers: int = 4,
    tokenizer_name: str = "gpt2",
    window_size: int = 64,
    kmer_len: int = 4,
    max_texts: int = 1_000_000,
    streaming: bool = False,
    verbose: bool = True,
) -> DataLoader:
    """
    Create a DataLoader for MiniPile training.
    
    Args:
        batch_size: Batch size.
        num_workers: Number of data loading workers.
        tokenizer_name: HuggingFace tokenizer name.
        window_size: Tokens per window.
        kmer_len: K-mer length to reconstruct.
        max_texts: Maximum texts to load.
        streaming: Use streaming mode (more memory efficient).
        verbose: Print progress.
    
    Returns:
        DataLoader for training.
    """
    if streaming:
        if verbose:
            print("Using streaming MiniPile dataset...")
        dataset = StreamingMiniPileDataset(
            tokenizer_name=tokenizer_name,
            window_size=window_size,
            kmer_len=kmer_len,
            max_texts=max_texts,
        )
    else:
        texts = load_minipile(split="train", max_texts=max_texts, verbose=verbose)
        dataset = MiniPileDataset(
            texts=texts,
            tokenizer_name=tokenizer_name,
            window_size=window_size,
            kmer_len=kmer_len,
        )
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
    )
    
    return dataloader


def get_minipile_stats(verbose: bool = True) -> Dict:
    """
    Get statistics about MiniPile dataset.
    """
    if verbose:
        print("Loading MiniPile for statistics...")
    
    dataset = load_dataset(
        "JeanKaddour/minipile",
        split="train",
        trust_remote_code=True,
    )
    
    total_chars = 0
    total_texts = 0
    length_buckets = {
        "<100": 0,
        "100-500": 0,
        "500-1000": 0,
        "1000-5000": 0,
        "5000+": 0,
    }
    
    for example in dataset:
        text = example.get("text", "")
        total_texts += 1
        total_chars += len(text)
        
        length = len(text)
        if length < 100:
            length_buckets["<100"] += 1
        elif length < 500:
            length_buckets["100-500"] += 1
        elif length < 1000:
            length_buckets["500-1000"] += 1
        elif length < 5000:
            length_buckets["1000-5000"] += 1
        else:
            length_buckets["5000+"] += 1
    
    stats = {
        "total_texts": total_texts,
        "total_chars": total_chars,
        "avg_chars_per_text": total_chars / total_texts if total_texts > 0 else 0,
        "length_distribution": length_buckets,
    }
    
    if verbose:
        print("\n" + "="*60)
        print("MINIPILE STATISTICS")
        print("="*60)
        print(f"Total texts: {stats['total_texts']:,}")
        print(f"Total characters: {stats['total_chars']:,}")
        print(f"Avg chars/text: {stats['avg_chars_per_text']:,.0f}")
        print("\nLength distribution:")
        for bucket, count in length_buckets.items():
            pct = 100 * count / total_texts if total_texts > 0 else 0
            print(f"  {bucket:12s}: {count:8,} ({pct:5.1f}%)")
    
    return stats


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Test MiniPile data loading")
    parser.add_argument("--stats", action="store_true", help="Print dataset statistics")
    parser.add_argument("--test", action="store_true", help="Test dataloader")
    parser.add_argument("--streaming", action="store_true", help="Use streaming mode")
    args = parser.parse_args()
    
    if args.stats:
        get_minipile_stats(verbose=True)
    
    if args.test or not args.stats:
        print("\nTesting MiniPile dataloader...")
        dl = make_minipile_dataloader(
            batch_size=4,
            num_workers=0,
            max_texts=1000,
            streaming=args.streaming,
            verbose=True,
        )
        
        for i, batch in enumerate(dl):
            print(f"\nBatch {i}:")
            print(f"  tokens shape: {batch['tokens'].shape}")
            print(f"  kmer_start: {batch['kmer_start']}")
            print(f"  kmer_len: {batch['kmer_len']}")
            
            if i >= 2:
                break
        
        print("\nMiniPile dataloader test complete!")
