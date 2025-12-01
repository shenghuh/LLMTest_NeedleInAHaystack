# VQ-VAE Training Pipeline with Curriculum Learning and LoRA

This document describes the complete training pipeline for the k-mer VQ-VAE coarse semantic filter.

## Overview

The training uses a **curriculum learning** approach with **LoRA adapters** for efficient domain adaptation:

```
Phase 1a: Web Text        →  Phase 1b: Domain LoRA    →  Phase 2: Query-Retrieval Alignment
(stable base training)       (domain adaptation)          (contrastive alignment)
```

## Architecture Summary

| Component | Value |
|-----------|-------|
| Model dimension | 384 |
| Layers | 4 |
| Attention heads | 4 |
| Codebook size | 1024 |
| K-mer length | 4 |
| Window size | 64 |
| Teacher | all-MiniLM-L6-v2 |

---

## Phase 1a: Base Training on Web Text

**Purpose**: Train stable VQ-VAE codebook on clean, homogeneous web text.

**Data**:
- WikiText-103 (~1.8M texts)
- C4 (200k documents, streaming)

**Key hyperparameters**:
- `num_codes=1024`
- `diversity_weight=0.2`
- `ema_decay=0.99`
- `max_tokens_total=200_000_000`
- Dead code restart enabled

**Run**:
```bash
cd localtesting/vq-filter
python vqvae_phase1_train.py
```

**Expected outcomes**:
- Perplexity: ~900-1000 (93%+ codebook utilization)
- Codebook usage should be diverse (check wandb histograms)
- No collapse (perplexity staying >> 1)

**Output**: `kmer_vqvae_phase1.pt`

---

## Phase 1b: Domain Adaptation with LoRA

**Purpose**: Fine-tune for academic papers and conversational text using LoRA (efficient adaptation without forgetting).

**Data**:
- **Academic**: arXiv abstracts, PubMed abstracts
- **Conversational**: OpenAssistant, UltraChat, Alpaca, HH-RLHF, ShareGPT, LMSYS

**Key hyperparameters**:
- `lora_rank=8`
- `lora_alpha=16`
- Learning rate: `1e-4` (LoRA), `1e-5` (codebook, optional)
- Codebook frozen or fine-tuned with very low LR

**Run**:
```bash
python vqvae_phase1b_train_lora.py --phase1a_checkpoint kmer_vqvae_phase1.pt
```

**What happens**:
1. Loads Phase 1a checkpoint
2. Injects LoRA into encoder attention (Q, V projections)
3. Freezes base model, trains only LoRA + optional codebook
4. Uses mixed domain data

**Expected outcomes**:
- Maintains high perplexity from Phase 1a
- Adapts to domain-specific vocabulary patterns
- LoRA params: ~50k (vs ~2M total) = 2.5% trainable

**Output**: `kmer_vqvae_phase1b_lora.pt`

---

## Phase 2: Query-Retrieval Alignment

**Purpose**: Train projection adapters so queries and relevant passages map to similar codes.

**Data** (query-passage pairs):
- MS MARCO (web search)
- Natural Questions
- SQuAD
- HotpotQA
- TriviaQA
- ELI5

**Key hyperparameters**:
- `adapter_type="dual"` (separate query/passage adapters)
- `temperature=0.07` (InfoNCE)
- Learning rate: `3e-4`
- Base model and codebook frozen

**Run**:
```bash
python vqvae_phase2_train_adapters.py --checkpoint kmer_vqvae_phase1b_lora.pt
# Or directly from Phase 1a:
python vqvae_phase2_train_adapters.py --checkpoint kmer_vqvae_phase1.pt
```

**What happens**:
1. Loads Phase 1a or 1b checkpoint
2. Adds projection adapters (QueryAdapter, PassageAdapter)
3. Freezes everything except adapters
4. Contrastive learning on (query, passage) pairs

**Metrics to watch**:
- `q2p_acc` / `p2q_acc`: Retrieval accuracy (in-batch)
- `code_match_rate`: How often Q and P get same code
- `pos_similarity`: Cosine similarity of positive pairs

**Expected outcomes**:
- Retrieval accuracy: >80% (in-batch)
- Code match rate: ~30-50% (not 100% - codes are coarse)
- Positive similarity: >0.5

**Output**: `kmer_vqvae_phase2_aligned.pt`

---

## File Reference

| File | Purpose |
|------|---------|
| `vqvae_phase1_model.py` | Core VQ-VAE architecture |
| `vqvae_phase1_train.py` | Phase 1a training script |
| `vqvae_phase1a_data_webtext.py` | Web text data (WikiText-103, C4) |
| `vqvae_phase1b_data_domains.py` | Domain data (academic + conversational) |
| `vqvae_phase1b_train_lora.py` | Phase 1b LoRA fine-tuning |
| `lora_adapters.py` | LoRA + projection adapter implementations |
| `vqvae_phase2_data.py` | Query-passage pairs data |
| `vqvae_phase2_train_adapters.py` | Phase 2 alignment training |

---

## Quick Start

```bash
# 1. Phase 1a: Train base model (takes ~4-8 hours on GPU)
python vqvae_phase1_train.py

# 2. Phase 1b: Domain adaptation with LoRA (takes ~2-4 hours)
python vqvae_phase1b_train_lora.py --phase1a_checkpoint kmer_vqvae_phase1.pt

# 3. Phase 2: Query-retrieval alignment (takes ~2-4 hours)
python vqvae_phase2_train_adapters.py --checkpoint kmer_vqvae_phase1b_lora.pt
```

---

## Inference Usage

After training, use the model for filtering:

```python
import torch
from vqvae_phase1_model import KmerVQVAE
from lora_adapters import QueryPassageAdapters

# Load trained model
checkpoint = torch.load("kmer_vqvae_phase2_aligned.pt")
model = KmerVQVAE(**checkpoint["base_config"])
model.load_state_dict(checkpoint["base_model_state_dict"])

adapters = QueryPassageAdapters(d_model=384)
adapters.load_state_dict(checkpoint["adapter_state_dict"])

# Get codes for query and passages
query_code = model.encode(query_tokens, ...)["indices"]
passage_codes = model.encode(passage_tokens, ...)["indices"]

# Filter: keep passages with matching codes
matches = (passage_codes == query_code)
```

---

## Troubleshooting

### Codebook Collapse (Phase 1a)
- **Symptom**: Perplexity drops to 1-10
- **Fix**: Increase `diversity_weight`, ensure dead code restart is enabled

### Poor Domain Adaptation (Phase 1b)
- **Symptom**: High loss on domain data
- **Fix**: Increase LoRA rank, allow codebook fine-tuning

### Low Retrieval Accuracy (Phase 2)
- **Symptom**: q2p_acc < 50%
- **Fix**: Increase adapter hidden dim, lower temperature, more training steps

### Code Match Rate Too Low
- **Symptom**: <10% match rate
- **Fix**: This is expected for coarse filtering. Codes capture broad semantics, not exact matches.
