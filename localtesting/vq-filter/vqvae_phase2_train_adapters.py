"""
Phase 2 Training: Query-Retrieval Alignment with Adapters

Fine-tunes the VQ-VAE for query-passage alignment using projection adapters.
The base model and codebook are frozen; only adapter parameters are trained.

This uses a contrastive learning objective to make queries and their relevant
passages map to similar representations (and eventually similar codes).

Usage:
    python vqvae_phase2_train_adapters.py --checkpoint kmer_vqvae_phase1b_lora.pt
"""

import argparse
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from dataclasses import dataclass, asdict
from transformers import AutoTokenizer
from tqdm import tqdm
import wandb
from typing import Optional, Dict

from vqvae_phase2_data import make_phase2_dataloader
from vqvae_phase1_model import KmerVQVAE
from lora_adapters import (
    QueryPassageAdapters,
    ProjectionAdapter,
    freeze_base_model,
    count_parameters,
)


@dataclass
class Phase2Config:
    """Configuration for Phase 2 query-retrieval alignment."""
    
    # Checkpoint to load (Phase 1a or 1b)
    checkpoint: str = "kmer_vqvae_phase1.pt"
    
    # Training
    num_steps: int = 30_000
    batch_size: int = 128
    lr: float = 3e-4
    grad_clip: float = 1.0
    
    # Adapter settings
    adapter_type: str = "dual"  # "dual" (separate Q/P), "shared", or "single"
    adapter_hidden_dim: int = 384  # Hidden dim for adapters
    adapter_dropout: float = 0.1
    
    # Loss settings
    temperature: float = 0.07  # Temperature for InfoNCE
    use_hard_negatives: bool = False  # Use in-batch hard negatives
    
    # What to train
    train_encoder: bool = False  # Keep encoder frozen
    train_codebook: bool = False  # Keep codebook frozen
    
    # Data
    tokenizer_name: str = "gpt2"
    window_size: int = 64
    kmer_len: int = 4
    max_pairs: int = 500_000
    
    # Logging
    log_every: int = 100
    eval_every: int = 1000
    eval_steps: int = 50
    
    # Paths
    save_path: str = "kmer_vqvae_phase2_aligned.pt"
    wandb_project: str = "kmer-vqvae"
    wandb_run_name: str = "phase2-alignment"
    
    # Device
    device: str = "cuda"
    
    def to_dict(self):
        return asdict(self)


class AlignedVQVAE(nn.Module):
    """
    VQ-VAE with query-passage alignment adapters.
    
    Wraps the base VQ-VAE and adds adapters for query/passage transformation.
    """
    
    def __init__(
        self,
        base_model: KmerVQVAE,
        adapter_type: str = "dual",
        adapter_hidden_dim: int = 384,
        adapter_dropout: float = 0.1,
    ):
        super().__init__()
        self.base_model = base_model
        self.adapter_type = adapter_type
        
        d_model = base_model.d_model
        
        if adapter_type == "dual":
            # Separate adapters for query and passage
            self.adapters = QueryPassageAdapters(
                d_model=d_model,
                hidden_dim=adapter_hidden_dim,
                shared_base=False,
                dropout=adapter_dropout,
            )
        elif adapter_type == "shared":
            # Shared base, separate heads
            self.adapters = QueryPassageAdapters(
                d_model=d_model,
                hidden_dim=adapter_hidden_dim,
                shared_base=True,
                dropout=adapter_dropout,
            )
        elif adapter_type == "single":
            # Single adapter for both (simplest)
            self.adapters = ProjectionAdapter(
                d_model=d_model,
                hidden_dim=adapter_hidden_dim,
                dropout=adapter_dropout,
            )
        else:
            raise ValueError(f"Unknown adapter_type: {adapter_type}")
    
    def encode(self, tokens, kmer_start, kmer_len, is_query: bool = False):
        """
        Encode tokens and apply appropriate adapter.
        
        Returns:
            z_e: Encoder output (before VQ)
            z_q: Quantized output (after VQ)
            indices: Code indices
            z_adapted: Adapter-transformed representation
        """
        # Get base model encoding
        output = self.base_model(tokens, kmer_start, kmer_len)
        z_e = output["z_e"]
        z_q = output["z_q"]
        indices = output["indices"]
        
        # Apply adapter
        if self.adapter_type == "single":
            z_adapted = self.adapters(z_e)
        elif is_query:
            z_adapted = self.adapters.forward_query(z_e)
        else:
            z_adapted = self.adapters.forward_passage(z_e)
        
        return {
            "z_e": z_e,
            "z_q": z_q,
            "indices": indices,
            "z_adapted": z_adapted,
            "logits": output["logits"],
            "vq_loss": output["vq_loss"],
            "perplexity": output["perplexity"],
        }
    
    def forward(self, tokens, kmer_start, kmer_len):
        """Standard forward pass (for reconstruction)."""
        return self.base_model(tokens, kmer_start, kmer_len)


def compute_contrastive_loss(
    query_z: torch.Tensor,
    passage_z: torch.Tensor,
    temperature: float = 0.07,
) -> Dict[str, torch.Tensor]:
    """
    Compute InfoNCE contrastive loss.
    
    Args:
        query_z: Query representations [B, D]
        passage_z: Passage representations [B, D]
        temperature: Temperature for softmax
    
    Returns:
        Dictionary with loss and metrics
    """
    B = query_z.shape[0]
    
    # Normalize
    query_z = F.normalize(query_z, dim=-1)
    passage_z = F.normalize(passage_z, dim=-1)
    
    # Similarity matrix [B, B]
    sim_matrix = torch.mm(query_z, passage_z.t()) / temperature
    
    # Labels: diagonal is positive
    labels = torch.arange(B, device=query_z.device)
    
    # Symmetric loss
    loss_q2p = F.cross_entropy(sim_matrix, labels)
    loss_p2q = F.cross_entropy(sim_matrix.t(), labels)
    loss = (loss_q2p + loss_p2q) / 2
    
    # Metrics
    with torch.no_grad():
        # Retrieval accuracy
        q2p_acc = (sim_matrix.argmax(dim=1) == labels).float().mean()
        p2q_acc = (sim_matrix.argmax(dim=0) == labels).float().mean()
        
        # Mean cosine similarity of positive pairs
        pos_sim = (query_z * passage_z).sum(dim=-1).mean()
    
    return {
        "loss": loss,
        "q2p_acc": q2p_acc,
        "p2q_acc": p2q_acc,
        "pos_similarity": pos_sim,
    }


def compute_code_alignment_metrics(
    query_indices: torch.Tensor,
    passage_indices: torch.Tensor,
) -> Dict[str, float]:
    """Compute metrics for code alignment."""
    # Exact match rate
    match_rate = (query_indices == passage_indices).float().mean().item()
    
    return {
        "code_match_rate": match_rate,
    }


@torch.no_grad()
def evaluate(model, val_loader, device, config, eval_steps):
    """Evaluate alignment on validation set."""
    model.eval()
    
    total_loss = 0.0
    total_q2p_acc = 0.0
    total_p2q_acc = 0.0
    total_code_match = 0.0
    total_pos_sim = 0.0
    count = 0
    
    val_iter = iter(val_loader)
    
    for _ in range(eval_steps):
        try:
            batch = next(val_iter)
        except StopIteration:
            val_iter = iter(val_loader)
            batch = next(val_iter)
        
        query_tokens = batch["query_tokens"].to(device)
        query_kmer_start = batch["query_kmer_start"].to(device)
        passage_tokens = batch["passage_tokens"].to(device)
        passage_kmer_start = batch["passage_kmer_start"].to(device)
        kmer_len = batch["kmer_len"][0].item()
        
        # Encode
        query_out = model.encode(query_tokens, query_kmer_start, kmer_len, is_query=True)
        passage_out = model.encode(passage_tokens, passage_kmer_start, kmer_len, is_query=False)
        
        # Contrastive loss
        contrast = compute_contrastive_loss(
            query_out["z_adapted"],
            passage_out["z_adapted"],
            config.temperature,
        )
        
        # Code alignment
        code_metrics = compute_code_alignment_metrics(
            query_out["indices"],
            passage_out["indices"],
        )
        
        total_loss += contrast["loss"].item()
        total_q2p_acc += contrast["q2p_acc"].item()
        total_p2q_acc += contrast["p2q_acc"].item()
        total_pos_sim += contrast["pos_similarity"].item()
        total_code_match += code_metrics["code_match_rate"]
        count += 1
    
    model.train()
    
    return {
        "val_loss": total_loss / count,
        "val_q2p_acc": total_q2p_acc / count,
        "val_p2q_acc": total_p2q_acc / count,
        "val_retrieval_acc": (total_q2p_acc + total_p2q_acc) / (2 * count),
        "val_pos_similarity": total_pos_sim / count,
        "val_code_match_rate": total_code_match / count,
    }


def train_phase2(config: Phase2Config):
    """Main Phase 2 training loop."""
    
    device = torch.device(config.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # ==========================================================================
    # Load checkpoint
    # ==========================================================================
    print(f"\nLoading checkpoint: {config.checkpoint}")
    
    if not os.path.exists(config.checkpoint):
        raise FileNotFoundError(f"Checkpoint not found: {config.checkpoint}")
    
    checkpoint = torch.load(config.checkpoint, map_location=device)
    
    # Get base config (could be from Phase 1a or 1b)
    base_config = checkpoint.get("phase1a_config", checkpoint.get("config", {}))
    
    # Create base model
    base_model = KmerVQVAE(
        vocab_size=base_config.get("vocab_size", 50257),
        d_model=base_config.get("d_model", 384),
        n_layers=base_config.get("n_layers", 4),
        n_heads=base_config.get("n_heads", 4),
        num_codes=base_config.get("num_codes", 1024),
        commitment_cost=0.25,
        diversity_weight=0.0,  # Disable for Phase 2
        ema_decay=base_config.get("ema_decay", 0.99),
        dead_code_threshold=1.0,
        dead_code_restart_prob=0.0,  # Disable restarts
    ).to(device)
    
    # Load weights
    base_model.load_state_dict(checkpoint["model_state_dict"], strict=False)
    print(f"  Loaded base model from step {checkpoint.get('step', 'unknown')}")
    
    # Create aligned model with adapters
    model = AlignedVQVAE(
        base_model=base_model,
        adapter_type=config.adapter_type,
        adapter_hidden_dim=config.adapter_hidden_dim,
        adapter_dropout=config.adapter_dropout,
    ).to(device)
    
    # Freeze base model
    for param in model.base_model.parameters():
        param.requires_grad = False
    
    # Only adapters are trainable
    for param in model.adapters.parameters():
        param.requires_grad = True
    
    # Count parameters
    param_counts = count_parameters(model)
    print(f"  Total parameters: {param_counts['total']:,}")
    print(f"  Trainable (adapters): {param_counts['trainable']:,} ({param_counts['trainable_pct']:.2f}%)")
    
    # ==========================================================================
    # Setup data
    # ==========================================================================
    tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    print("\nLoading query-passage data...")
    
    train_loader = make_phase2_dataloader(
        batch_size=config.batch_size,
        num_workers=4,
        tokenizer_name=config.tokenizer_name,
        window_size=config.window_size,
        kmer_len=config.kmer_len,
        max_pairs=config.max_pairs,
        verbose=True,
        split="train",
    )
    
    val_loader = make_phase2_dataloader(
        batch_size=config.batch_size,
        num_workers=2,
        tokenizer_name=config.tokenizer_name,
        window_size=config.window_size,
        kmer_len=config.kmer_len,
        max_pairs=config.max_pairs // 10,
        verbose=True,
        split="val",
    )
    
    train_iter = iter(train_loader)
    
    # ==========================================================================
    # Optimizer
    # ==========================================================================
    optimizer = optim.AdamW(
        model.adapters.parameters(),
        lr=config.lr,
        weight_decay=0.01,
    )
    
    # Cosine schedule with warmup
    warmup_steps = min(1000, config.num_steps // 10)
    
    def lr_lambda(step):
        if step < warmup_steps:
            return step / warmup_steps
        progress = (step - warmup_steps) / (config.num_steps - warmup_steps)
        return 0.5 * (1 + torch.cos(torch.tensor(progress * 3.14159)).item())
    
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    # ==========================================================================
    # Initialize wandb
    # ==========================================================================
    wandb.init(
        project=config.wandb_project,
        name=config.wandb_run_name,
        config=config.to_dict(),
    )
    
    # ==========================================================================
    # Training loop
    # ==========================================================================
    print(f"\nStarting Phase 2 alignment training for {config.num_steps} steps...")
    print(f"  Adapter type: {config.adapter_type}")
    print(f"  Temperature: {config.temperature}")
    
    model.train()
    pbar = tqdm(range(1, config.num_steps + 1), desc="Phase 2 Training")
    
    # Metrics accumulators
    metrics_accum = {
        "loss": 0.0,
        "q2p_acc": 0.0,
        "p2q_acc": 0.0,
        "pos_similarity": 0.0,
        "code_match_rate": 0.0,
    }
    accum_count = 0
    
    for step in pbar:
        optimizer.zero_grad()
        
        # Get batch
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            batch = next(train_iter)
        
        query_tokens = batch["query_tokens"].to(device)
        query_kmer_start = batch["query_kmer_start"].to(device)
        passage_tokens = batch["passage_tokens"].to(device)
        passage_kmer_start = batch["passage_kmer_start"].to(device)
        kmer_len = batch["kmer_len"][0].item()
        
        # Encode query and passage
        query_out = model.encode(query_tokens, query_kmer_start, kmer_len, is_query=True)
        passage_out = model.encode(passage_tokens, passage_kmer_start, kmer_len, is_query=False)
        
        # Contrastive loss
        contrast = compute_contrastive_loss(
            query_out["z_adapted"],
            passage_out["z_adapted"],
            config.temperature,
        )
        
        loss = contrast["loss"]
        
        # Backward
        loss.backward()
        
        if config.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.adapters.parameters(), config.grad_clip)
        
        optimizer.step()
        scheduler.step()
        
        # Code alignment metrics (no grad)
        with torch.no_grad():
            code_metrics = compute_code_alignment_metrics(
                query_out["indices"],
                passage_out["indices"],
            )
        
        # Accumulate metrics
        metrics_accum["loss"] += loss.item()
        metrics_accum["q2p_acc"] += contrast["q2p_acc"].item()
        metrics_accum["p2q_acc"] += contrast["p2q_acc"].item()
        metrics_accum["pos_similarity"] += contrast["pos_similarity"].item()
        metrics_accum["code_match_rate"] += code_metrics["code_match_rate"]
        accum_count += 1
        
        pbar.set_postfix({
            "loss": f"{loss.item():.4f}",
            "acc": f"{(contrast['q2p_acc'].item() + contrast['p2q_acc'].item())/2:.2%}",
            "match": f"{code_metrics['code_match_rate']:.2%}",
        })
        
        # Logging
        if step % config.log_every == 0:
            avg_metrics = {k: v / accum_count for k, v in metrics_accum.items()}
            avg_metrics["learning_rate"] = scheduler.get_last_lr()[0]
            avg_metrics["retrieval_acc"] = (avg_metrics["q2p_acc"] + avg_metrics["p2q_acc"]) / 2
            avg_metrics["step"] = step
            
            wandb.log(avg_metrics)
            
            for k in metrics_accum:
                metrics_accum[k] = 0.0
            accum_count = 0
        
        # Evaluation
        if step % config.eval_every == 0:
            print(f"\n\nStep {step}: Running evaluation...")
            
            val_metrics = evaluate(model, val_loader, device, config, config.eval_steps)
            
            print(f"  Val retrieval acc: {val_metrics['val_retrieval_acc']:.2%}")
            print(f"  Val code match rate: {val_metrics['val_code_match_rate']:.2%}")
            print(f"  Val positive similarity: {val_metrics['val_pos_similarity']:.4f}")
            
            val_metrics["step"] = step
            wandb.log(val_metrics)
            
            # Save checkpoint
            save_checkpoint(model, optimizer, step, config, base_config, val_metrics)
    
    # Final save
    print("\n" + "="*60)
    print("Phase 2 training complete!")
    print("="*60)
    
    save_checkpoint(model, optimizer, config.num_steps, config, base_config, {})
    
    wandb.finish()


def save_checkpoint(model, optimizer, step, config, base_config, val_metrics):
    """Save checkpoint."""
    torch.save({
        "step": step,
        "model_state_dict": model.state_dict(),
        "base_model_state_dict": model.base_model.state_dict(),
        "adapter_state_dict": model.adapters.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "config": config.to_dict(),
        "base_config": base_config,
        "val_metrics": val_metrics,
    }, config.save_path)
    print(f"  Saved checkpoint to {config.save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phase 2: Query-Retrieval Alignment")
    parser.add_argument("--checkpoint", type=str, default="kmer_vqvae_phase1.pt",
                        help="Path to Phase 1 checkpoint")
    parser.add_argument("--num_steps", type=int, default=30000,
                        help="Number of training steps")
    parser.add_argument("--batch_size", type=int, default=128,
                        help="Batch size")
    parser.add_argument("--lr", type=float, default=3e-4,
                        help="Learning rate")
    parser.add_argument("--adapter_type", type=str, default="dual",
                        choices=["dual", "shared", "single"],
                        help="Adapter architecture type")
    parser.add_argument("--temperature", type=float, default=0.07,
                        help="Temperature for contrastive loss")
    parser.add_argument("--save_path", type=str, default="kmer_vqvae_phase2_aligned.pt",
                        help="Path to save checkpoint")
    
    args = parser.parse_args()
    
    config = Phase2Config(
        checkpoint=args.checkpoint,
        num_steps=args.num_steps,
        batch_size=args.batch_size,
        lr=args.lr,
        adapter_type=args.adapter_type,
        temperature=args.temperature,
        save_path=args.save_path,
        wandb_run_name=f"phase2-{args.adapter_type}-t{args.temperature}",
    )
    
    train_phase2(config)
