"""
Phase 2 Data Loading: Query-Passage Pairs for Retrieval Alignment

This module loads datasets with (query, positive_passage) pairs for training
the VQ-VAE to map semantically related queries and passages to the same codes.

Datasets used:
- MS MARCO: Web search queries + relevant passages
- Natural Questions: Wikipedia questions + answer passages  
- HotpotQA: Multi-hop reasoning questions + supporting passages
- SQuAD: Reading comprehension questions + context passages
"""

import random
import warnings
from typing import List, Dict, Tuple, Optional, Iterator
from dataclasses import dataclass

import torch
from torch import LongTensor
from torch.utils.data import DataLoader, IterableDataset
from datasets import load_dataset
from transformers import AutoTokenizer

# Suppress dataset loading warnings
warnings.filterwarnings("ignore", message=".*trust_remote_code.*")
warnings.filterwarnings("ignore", message=".*loading script.*")


@dataclass
class QueryPassagePair:
    """A query-passage pair for retrieval alignment."""
    query: str
    positive_passage: str
    source: str  # Dataset source for debugging


def load_query_passage_pairs(
    verbose: bool = True,
    split: str = "train",
    val_ratio: float = 0.05,
    seed: int = 42,
    max_pairs_per_source: int = 100_000,
) -> List[QueryPassagePair]:
    """
    Load query-passage pairs from multiple retrieval datasets.
    
    Args:
        verbose: Whether to print loading progress.
        split: "train" or "val" - which split to return.
        val_ratio: Fraction of data to use for validation.
        seed: Random seed for reproducible train/val split.
        max_pairs_per_source: Max pairs to load from each dataset.
    
    Returns:
        List of QueryPassagePair objects.
    """
    all_pairs = []
    source_counts = {}
    
    # ==========================================================================
    # MS MARCO Passages
    # ==========================================================================
    try:
        if verbose:
            print("Loading MS MARCO passages...")
        msmarco = load_dataset(
            "ms_marco", "v2.1", 
            split="train", 
            trust_remote_code=True
        )
        
        msmarco_pairs = []
        for ex in msmarco:
            query = ex.get("query", "")
            passages = ex.get("passages", {})
            
            if not query.strip():
                continue
                
            # Get passages marked as relevant (is_selected=1)
            passage_texts = passages.get("passage_text", [])
            is_selected = passages.get("is_selected", [])
            
            for text, selected in zip(passage_texts, is_selected):
                if selected == 1 and text.strip():
                    msmarco_pairs.append(QueryPassagePair(
                        query=query.strip(),
                        positive_passage=text.strip(),
                        source="msmarco"
                    ))
                    break  # One positive per query
            
            if len(msmarco_pairs) >= max_pairs_per_source:
                break
        
        all_pairs.extend(msmarco_pairs)
        source_counts["msmarco"] = len(msmarco_pairs)
        if verbose:
            print(f"  Loaded {len(msmarco_pairs)} pairs from MS MARCO")
    except Exception as e:
        if verbose:
            print(f"  Could not load MS MARCO: {e}")
    
    # ==========================================================================
    # Natural Questions
    # ==========================================================================
    try:
        if verbose:
            print("Loading Natural Questions...")
        nq = load_dataset(
            "natural_questions", "default",
            split="train",
            trust_remote_code=True
        )
        
        nq_pairs = []
        for ex in nq:
            question = ex.get("question", {}).get("text", "")
            
            # Get document context (long answer candidates)
            doc = ex.get("document", {})
            doc_text = doc.get("tokens", [])
            
            # Get short answer if available
            annotations = ex.get("annotations", [])
            
            if not question.strip() or not doc_text:
                continue
            
            # Join document tokens for passage
            if isinstance(doc_text, list):
                passage = " ".join(doc_text[:500])  # Limit length
            else:
                passage = str(doc_text)[:2000]
            
            if passage.strip():
                nq_pairs.append(QueryPassagePair(
                    query=question.strip(),
                    positive_passage=passage.strip(),
                    source="natural_questions"
                ))
            
            if len(nq_pairs) >= max_pairs_per_source:
                break
        
        all_pairs.extend(nq_pairs)
        source_counts["natural_questions"] = len(nq_pairs)
        if verbose:
            print(f"  Loaded {len(nq_pairs)} pairs from Natural Questions")
    except Exception as e:
        if verbose:
            print(f"  Could not load Natural Questions: {e}")
    
    # ==========================================================================
    # SQuAD v2
    # ==========================================================================
    try:
        if verbose:
            print("Loading SQuAD v2...")
        squad = load_dataset("squad_v2", split="train", trust_remote_code=True)
        
        squad_pairs = []
        for ex in squad:
            question = ex.get("question", "")
            context = ex.get("context", "")
            
            # Skip unanswerable questions (empty answers)
            answers = ex.get("answers", {})
            answer_texts = answers.get("text", [])
            if not answer_texts:
                continue
            
            if question.strip() and context.strip():
                squad_pairs.append(QueryPassagePair(
                    query=question.strip(),
                    positive_passage=context.strip(),
                    source="squad"
                ))
            
            if len(squad_pairs) >= max_pairs_per_source:
                break
        
        all_pairs.extend(squad_pairs)
        source_counts["squad"] = len(squad_pairs)
        if verbose:
            print(f"  Loaded {len(squad_pairs)} pairs from SQuAD")
    except Exception as e:
        if verbose:
            print(f"  Could not load SQuAD: {e}")
    
    # ==========================================================================
    # HotpotQA
    # ==========================================================================
    try:
        if verbose:
            print("Loading HotpotQA...")
        hotpot = load_dataset(
            "hotpot_qa", "fullwiki",
            split="train",
            trust_remote_code=True
        )
        
        hotpot_pairs = []
        for ex in hotpot:
            question = ex.get("question", "")
            
            # Combine supporting facts into passage
            context = ex.get("context", {})
            titles = context.get("title", [])
            sentences = context.get("sentences", [])
            
            if not question.strip():
                continue
            
            # Flatten sentences from all supporting documents
            passage_parts = []
            for title, sents in zip(titles, sentences):
                if isinstance(sents, list):
                    passage_parts.extend(sents)
            
            passage = " ".join(passage_parts)
            
            if passage.strip():
                hotpot_pairs.append(QueryPassagePair(
                    query=question.strip(),
                    positive_passage=passage.strip()[:2000],  # Limit length
                    source="hotpotqa"
                ))
            
            if len(hotpot_pairs) >= max_pairs_per_source:
                break
        
        all_pairs.extend(hotpot_pairs)
        source_counts["hotpotqa"] = len(hotpot_pairs)
        if verbose:
            print(f"  Loaded {len(hotpot_pairs)} pairs from HotpotQA")
    except Exception as e:
        if verbose:
            print(f"  Could not load HotpotQA: {e}")
    
    # ==========================================================================
    # TriviaQA
    # ==========================================================================
    try:
        if verbose:
            print("Loading TriviaQA...")
        trivia = load_dataset(
            "trivia_qa", "rc",
            split="train",
            trust_remote_code=True
        )
        
        trivia_pairs = []
        for ex in trivia:
            question = ex.get("question", "")
            
            # Get search results or entity pages as context
            search_results = ex.get("search_results", {})
            search_contexts = search_results.get("search_context", [])
            
            entity_pages = ex.get("entity_pages", {})
            wiki_contexts = entity_pages.get("wiki_context", [])
            
            if not question.strip():
                continue
            
            # Use first available context
            passage = ""
            if search_contexts:
                passage = search_contexts[0] if search_contexts[0] else ""
            elif wiki_contexts:
                passage = wiki_contexts[0] if wiki_contexts[0] else ""
            
            if passage.strip():
                trivia_pairs.append(QueryPassagePair(
                    query=question.strip(),
                    positive_passage=passage.strip()[:2000],
                    source="triviaqa"
                ))
            
            if len(trivia_pairs) >= max_pairs_per_source:
                break
        
        all_pairs.extend(trivia_pairs)
        source_counts["triviaqa"] = len(trivia_pairs)
        if verbose:
            print(f"  Loaded {len(trivia_pairs)} pairs from TriviaQA")
    except Exception as e:
        if verbose:
            print(f"  Could not load TriviaQA: {e}")
    
    # ==========================================================================
    # ELI5 (Reddit Explain Like I'm 5)
    # ==========================================================================
    try:
        if verbose:
            print("Loading ELI5...")
        eli5 = load_dataset(
            "eli5", 
            split="train_eli5",
            trust_remote_code=True
        )
        
        eli5_pairs = []
        for ex in eli5:
            title = ex.get("title", "")
            selftext = ex.get("selftext", "")
            answers = ex.get("answers", {})
            answer_texts = answers.get("text", [])
            
            # Query is the question title
            query = title.strip()
            if not query:
                continue
            
            # Passage is the top answer
            if answer_texts:
                # Get highest scored answer
                scores = answers.get("score", [0] * len(answer_texts))
                if scores:
                    best_idx = scores.index(max(scores))
                    passage = answer_texts[best_idx]
                else:
                    passage = answer_texts[0]
                
                if passage.strip():
                    eli5_pairs.append(QueryPassagePair(
                        query=query,
                        positive_passage=passage.strip()[:2000],
                        source="eli5"
                    ))
            
            if len(eli5_pairs) >= max_pairs_per_source:
                break
        
        all_pairs.extend(eli5_pairs)
        source_counts["eli5"] = len(eli5_pairs)
        if verbose:
            print(f"  Loaded {len(eli5_pairs)} pairs from ELI5")
    except Exception as e:
        if verbose:
            print(f"  Could not load ELI5: {e}")
    
    # ==========================================================================
    # SUMMARY AND SPLIT
    # ==========================================================================
    if verbose:
        print("\n" + "="*60)
        print("QUERY-PASSAGE PAIRS SUMMARY")
        print("="*60)
        total = sum(source_counts.values())
        for source, count in source_counts.items():
            pct = 100.0 * count / total if total > 0 else 0
            print(f"  {source:20s}: {count:8d} ({pct:5.1f}%)")
        print(f"  {'TOTAL':20s}: {total:8d}")
    
    if verbose:
        print(f"\nTotal pairs loaded: {len(all_pairs)}")
    
    # Deterministic shuffle with seed for reproducible split
    rng = random.Random(seed)
    rng.shuffle(all_pairs)
    
    # Split into train and val
    n_val = int(len(all_pairs) * val_ratio)
    if split == "val":
        pairs = all_pairs[:n_val]
        if verbose:
            print(f"Using validation split: {len(pairs)} pairs")
    else:  # train
        pairs = all_pairs[n_val:]
        if verbose:
            print(f"Using training split: {len(pairs)} pairs")
    
    return pairs


class QueryPassageDataset(IterableDataset):
    """
    Dataset that yields tokenized (query, passage) pairs for contrastive learning.
    
    For each pair, we extract windows from both query and passage,
    which will be used to compute alignment loss.
    """
    
    def __init__(
        self,
        pairs: List[QueryPassagePair],
        tokenizer_name: str = "gpt2",
        window_size: int = 64,
        kmer_len: int = 4,
        max_pairs: int = 1_000_000,
        shuffle: bool = True,
        seed: int = 42,
    ):
        """
        Args:
            pairs: List of QueryPassagePair objects.
            tokenizer_name: HuggingFace tokenizer name.
            window_size: Size of each token window.
            kmer_len: Length of k-mer to reconstruct.
            max_pairs: Maximum number of pairs to use.
            shuffle: Whether to shuffle pairs each epoch.
            seed: Random seed for shuffling.
        """
        super().__init__()
        self.pairs = pairs[:max_pairs]
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
    
    def _tokenize_and_window(self, text: str) -> Optional[Dict[str, torch.Tensor]]:
        """Tokenize text and extract a random window with k-mer position."""
        encoded = self.tokenizer.encode(
            text,
            add_special_tokens=False,
            truncation=False,
        )
        
        if len(encoded) < self.window_size:
            # Pad short sequences
            pad_len = self.window_size - len(encoded)
            encoded = encoded + [self.pad_token_id] * pad_len
        
        # Random window start
        max_start = len(encoded) - self.window_size
        if max_start > 0:
            start = random.randint(0, max_start)
        else:
            start = 0
        
        window = encoded[start : start + self.window_size]
        
        # Pick a central k-mer start index
        margin = self.window_size // 4
        min_kmer_start = margin
        max_kmer_start = self.window_size - margin - self.kmer_len
        
        if max_kmer_start <= min_kmer_start:
            kmer_start = (self.window_size - self.kmer_len) // 2
        else:
            kmer_start = random.randint(min_kmer_start, max_kmer_start)
        
        return {
            "tokens": LongTensor(window),
            "kmer_start": LongTensor([kmer_start]).squeeze(),
            "kmer_len": LongTensor([self.kmer_len]).squeeze(),
        }
    
    def __iter__(self) -> Iterator[Dict[str, torch.Tensor]]:
        pairs = self.pairs.copy()
        
        if self.shuffle:
            rng = random.Random(self.seed)
            rng.shuffle(pairs)
        
        for pair in pairs:
            query_data = self._tokenize_and_window(pair.query)
            passage_data = self._tokenize_and_window(pair.positive_passage)
            
            if query_data is None or passage_data is None:
                continue
            
            yield {
                "query_tokens": query_data["tokens"],
                "query_kmer_start": query_data["kmer_start"],
                "passage_tokens": passage_data["tokens"],
                "passage_kmer_start": passage_data["kmer_start"],
                "kmer_len": query_data["kmer_len"],
            }


def make_phase2_dataloader(
    batch_size: int = 128,
    num_workers: int = 4,
    tokenizer_name: str = "gpt2",
    window_size: int = 64,
    kmer_len: int = 4,
    max_pairs: int = 500_000,
    verbose: bool = True,
    split: str = "train",
) -> DataLoader:
    """
    Create a DataLoader for Phase 2 query-passage alignment training.
    
    Args:
        batch_size: Batch size for the dataloader.
        num_workers: Number of worker processes.
        tokenizer_name: HuggingFace tokenizer name.
        window_size: Size of each token window.
        kmer_len: Length of k-mer to reconstruct.
        max_pairs: Maximum number of pairs to use.
        verbose: Whether to print loading progress.
        split: "train" or "val" - which data split to use.
    
    Returns:
        DataLoader yielding batches of query-passage pairs.
    """
    if verbose:
        print(f"Loading query-passage pairs ({split} split)...")
    
    pairs = load_query_passage_pairs(verbose=verbose, split=split)
    
    if verbose:
        print("Creating QueryPassageDataset...")
    
    dataset = QueryPassageDataset(
        pairs=pairs,
        tokenizer_name=tokenizer_name,
        window_size=window_size,
        kmer_len=kmer_len,
        max_pairs=max_pairs,
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
    print("Testing Phase 2 dataloader...")
    dl = make_phase2_dataloader(batch_size=4, num_workers=0, max_pairs=1000)
    
    for i, batch in enumerate(dl):
        print(f"\nBatch {i}:")
        print(f"  query_tokens shape: {batch['query_tokens'].shape}")
        print(f"  passage_tokens shape: {batch['passage_tokens'].shape}")
        print(f"  query_kmer_start: {batch['query_kmer_start']}")
        print(f"  passage_kmer_start: {batch['passage_kmer_start']}")
        if i >= 2:
            break
    
    print("\nPhase 2 data loading test complete!")
