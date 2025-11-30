"""
Phase 2 Training: Query-Retrieval Alignment

This phase fine-tunes the Phase 1 VQ-VAE to align query and passage representations.
The goal is to make queries and their relevant passages map to similar codes.

Training objectives:
1. Reconstruction loss (preserve Phase 1 capabilities)
2. Code alignment loss (query and positive passage should share codes)
3. Contrastive loss (push apart codes for unrelated query-passage pairs)

Usage:
    python vqvae_phase2_train.py --phase1_checkpoint kmer_vqvae_phase1.pt
"""

import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from dataclasses import dataclass, field, asdict
from transformers import AutoTokenizer
from tqdm import tqdm
import wandb
from typing import Optional, Tuple
import os

from vqvae_phase2_data import make_phase2_dataloader
from vqvae_phase1_data import make_dataloader as make_phase1_dataloader
from vqvae_phase1_model import KmerVQVAE


@dataclass
class Phase2Config:
    """Configuration for Phase 2 query-retrieval alignment training."""
    
    # Phase 1 checkpoint (required)
    phase1_checkpoint: str = "kmer_vqvae_phase1.pt"
    
    # Training
    num_steps: int = 30_000
    batch_size: int = 128
    lr: float = 1e-4  # Lower LR for fine-tuning
    grad_clip: float = 1.0
    
    # Loss weights
    recon_weight: float = 1.0  # Reconstruction loss weight
    align_weight: float = 0.5  # Code alignment loss weight
    contrastive_weight: float = 0.3  # Contrastive loss weight
    commitment_weight: float = 0.25  # VQ commitment loss
    
    # Alignment loss settings
    alignment_type: str = "soft"  # "hard" (same code) or "soft" (cosine similarity)
    temperature: float = 0.1  # Temperature for contrastive loss
    
    # Data
    tokenizer_name: str = "gpt2"
    window_size: int = 64
    kmer_len: int = 4
    max_pairs: int = 500_000
    
    # Mixed training: also include Phase 1 data to prevent forgetting
    phase1_mix_ratio: float = 0.3  # 30% Phase 1 batches, 70% Phase 2
    phase1_max_tokens: int = 50_000_000
    
    # Logging & evaluation
    log_every: int = 100
    eval_every: int = 1000
    eval_steps: int = 50
    
    # Paths & wandb
    save_path: str = "kmer_vqvae_phase2.pt"
    wandb_project: str = "kmer-vqvae"
    wandb_run_name: str = "phase2-alignment"
    
    # Device
    device: str = "cuda"
    
    # Freeze encoder during initial steps (optional)
    freeze_encoder_steps: int = 0  # Set >0 to freeze encoder initially
    
    def to_dict(self) -> dict:
        """Convert config to dictionary for wandb logging."""
        return asdict(self)


def compute_code_alignment_loss(
    query_indices: torch.Tensor,
    passage_indices: torch.Tensor,
    query_z_e: torch.Tensor,
    passage_z_e: torch.Tensor,
    alignment_type: str = "soft",
    temperature: float = 0.1,
) -> Tuple[torch.Tensor, dict]:
    """
    Compute alignment loss between query and passage codes.
    
    Args:
        query_indices: Code indices for query windows [B]
        passage_indices: Code indices for passage windows [B]
        query_z_e: Encoder output for queries [B, D]
        passage_z_e: Encoder output for passages [B, D]
        alignment_type: "hard" or "soft"
        temperature: Temperature for soft alignment
    
    Returns:
        Alignment loss and metrics dict
    """
    B = query_indices.shape[0]
    metrics = {}
    
    if alignment_type == "hard":
        # Hard alignment: queries and passages should get the same code
        # Use cross-entropy where target is passage code for query, and vice versa
        # This is tricky since codes are discrete - use straight-through estimator implicitly
        
        # Compute how often query and passage get the same code
        same_code = (query_indices == passage_indices).float()
        code_match_rate = same_code.mean()
        
        # Loss: negative reward for matching (we want to maximize matches)
        # Use a soft proxy via embedding distance
        query_z_norm = F.normalize(query_z_e, dim=-1)
        passage_z_norm = F.normalize(passage_z_e, dim=-1)
        
        # Cosine similarity should be high for aligned pairs
        cos_sim = (query_z_norm * passage_z_norm).sum(dim=-1)
        loss = (1 - cos_sim).mean()  # Push cosine similarity to 1
        
        metrics["code_match_rate"] = code_match_rate.item()
        metrics["mean_cos_sim"] = cos_sim.mean().item()
        
    else:  # soft alignment
        # Soft alignment: encoder outputs should be similar
        query_z_norm = F.normalize(query_z_e, dim=-1)
        passage_z_norm = F.normalize(passage_z_e, dim=-1)
        
        # InfoNCE-style contrastive loss
        # Positive pairs: (query_i, passage_i)
        # Negative pairs: (query_i, passage_j) for j != i
        
        # Compute similarity matrix [B, B]
        sim_matrix = torch.mm(query_z_norm, passage_z_norm.t()) / temperature
        
        # Labels: diagonal elements are positives
        labels = torch.arange(B, device=query_indices.device)
        
        # Cross-entropy loss (query to passage)
        loss_q2p = F.cross_entropy(sim_matrix, labels)
        
        # Cross-entropy loss (passage to query)
        loss_p2q = F.cross_entropy(sim_matrix.t(), labels)
        
        loss = (loss_q2p + loss_p2q) / 2
        
        # Metrics
        with torch.no_grad():
            same_code = (query_indices == passage_indices).float()
            cos_sim = (query_z_norm * passage_z_norm).sum(dim=-1)
            
            # Retrieval accuracy: is the correct passage in top-1?
            top1_q2p = (sim_matrix.argmax(dim=1) == labels).float().mean()
            top1_p2q = (sim_matrix.argmax(dim=0) == labels).float().mean()
            
        metrics["code_match_rate"] = same_code.mean().item()
        metrics["mean_cos_sim"] = cos_sim.mean().item()
        metrics["retrieval_acc_q2p"] = top1_q2p.item()
        metrics["retrieval_acc_p2q"] = top1_p2q.item()
    
    return loss, metrics


def compute_in_batch_contrastive_loss(
    query_z_e: torch.Tensor,
    passage_z_e: torch.Tensor,
    temperature: float = 0.1,
) -> torch.Tensor:
    """
    Standard in-batch contrastive loss (InfoNCE).
    
    Each query should be closer to its paired passage than to other passages in the batch.
    """
    B = query_z_e.shape[0]
    
    query_z_norm = F.normalize(query_z_e, dim=-1)
    passage_z_norm = F.normalize(passage_z_e, dim=-1)
    
    # Similarity matrix [B, B]
    sim_matrix = torch.mm(query_z_norm, passage_z_norm.t()) / temperature
    
    # Labels: diagonal is positive
    labels = torch.arange(B, device=query_z_e.device)
    
    # Symmetric loss
    loss = (F.cross_entropy(sim_matrix, labels) + F.cross_entropy(sim_matrix.t(), labels)) / 2
    
    return loss


@torch.no_grad()
def evaluate_phase2(
    model: KmerVQVAE,
    val_loader,
    device: torch.device,
    config: Phase2Config,
    eval_steps: int,
) -> dict:
    """Evaluate Phase 2 model on validation set."""
    model.eval()
    
    total_align_loss = 0.0
    total_code_match = 0.0
    total_cos_sim = 0.0
    total_retrieval_acc = 0.0
    count = 0
    
    val_iter = iter(val_loader)
    
    for _ in range(eval_steps):
        try:
            batch = next(val_iter)
        except StopIteration:
            val_iter = iter(val_loader)
            batch = next(val_iter)
        
        # Move to device
        query_tokens = batch["query_tokens"].to(device)
        query_kmer_start = batch["query_kmer_start"].to(device)
        passage_tokens = batch["passage_tokens"].to(device)
        passage_kmer_start = batch["passage_kmer_start"].to(device)
        kmer_len = batch["kmer_len"][0].item()
        
        # Forward pass for query
        query_output = model(query_tokens, query_kmer_start, kmer_len)
        query_z_e = query_output["z_e"]  # [B, D]
        query_indices = query_output["indices"]  # [B]
        
        # Forward pass for passage
        passage_output = model(passage_tokens, passage_kmer_start, kmer_len)
        passage_z_e = passage_output["z_e"]
        passage_indices = passage_output["indices"]
        
        # Compute alignment metrics
        _, metrics = compute_code_alignment_loss(
            query_indices, passage_indices,
            query_z_e, passage_z_e,
            alignment_type=config.alignment_type,
            temperature=config.temperature,
        )
        
        total_code_match += metrics["code_match_rate"]
        total_cos_sim += metrics["mean_cos_sim"]
        if "retrieval_acc_q2p" in metrics:
            total_retrieval_acc += (metrics["retrieval_acc_q2p"] + metrics["retrieval_acc_p2q"]) / 2
        count += 1
    
    model.train()
    
    return {
        "val_code_match_rate": total_code_match / count,
        "val_cos_sim": total_cos_sim / count,
        "val_retrieval_acc": total_retrieval_acc / count,
    }


def train_phase2(config: Phase2Config):
    """Main Phase 2 training loop."""
    
    device = torch.device(config.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Initialize wandb
    wandb.init(
        project=config.wandb_project,
        name=config.wandb_run_name,
        config=config.to_dict(),
    )
    
    # ==========================================================================
    # Load Phase 1 model
    # ==========================================================================
    print(f"\nLoading Phase 1 checkpoint: {config.phase1_checkpoint}")
    
    if not os.path.exists(config.phase1_checkpoint):
        raise FileNotFoundError(f"Phase 1 checkpoint not found: {config.phase1_checkpoint}")
    
    checkpoint = torch.load(config.phase1_checkpoint, map_location=device)
    phase1_config = checkpoint.get("config", {})
    
    # Extract model config from Phase 1
    model = KmerVQVAE(
        vocab_size=phase1_config.get("vocab_size", 50257),
        d_model=phase1_config.get("d_model", 384),
        n_layers=phase1_config.get("n_layers", 4),
        n_heads=phase1_config.get("n_heads", 4),
        num_codes=phase1_config.get("num_codes", 1024),
        commitment_cost=config.commitment_weight,
        diversity_weight=0.0,  # Disable diversity loss in Phase 2
        ema_decay=phase1_config.get("ema_decay", 0.99),
        dead_code_threshold=phase1_config.get("dead_code_threshold", 1.0),
        dead_code_restart_prob=0.0,  # Disable restarts in Phase 2
    ).to(device)
    
    # Load weights
    model.load_state_dict(checkpoint["model_state_dict"])
    print(f"  Loaded model with {sum(p.numel() for p in model.parameters()):,} parameters")
    print(f"  Phase 1 config: d_model={phase1_config.get('d_model')}, num_codes={phase1_config.get('num_codes')}")
    
    # Check Phase 1 metrics
    if "step" in checkpoint:
        print(f"  Phase 1 trained for {checkpoint['step']} steps")
    
    # ==========================================================================
    # Setup tokenizer and data loaders
    # ==========================================================================
    tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    vocab_size = tokenizer.vocab_size
    pad_token_id = tokenizer.pad_token_id
    
    print("\nCreating Phase 2 dataloaders...")
    
    # Phase 2 dataloader (query-passage pairs)
    train_loader_p2 = make_phase2_dataloader(
        batch_size=config.batch_size,
        num_workers=4,
        tokenizer_name=config.tokenizer_name,
        window_size=config.window_size,
        kmer_len=config.kmer_len,
        max_pairs=config.max_pairs,
        verbose=True,
        split="train",
    )
    
    val_loader_p2 = make_phase2_dataloader(
        batch_size=config.batch_size,
        num_workers=2,
        tokenizer_name=config.tokenizer_name,
        window_size=config.window_size,
        kmer_len=config.kmer_len,
        max_pairs=config.max_pairs // 10,
        verbose=True,
        split="val",
    )
    
    # Phase 1 dataloader (reconstruction, to prevent forgetting)
    if config.phase1_mix_ratio > 0:
        print("\nCreating Phase 1 dataloader for mixed training...")
        train_loader_p1 = make_phase1_dataloader(
            batch_size=config.batch_size,
            num_workers=2,
            tokenizer_name=config.tokenizer_name,
            window_size=config.window_size,
            kmer_len=config.kmer_len,
            max_tokens_total=config.phase1_max_tokens,
            verbose=True,
            split="train",
        )
        train_iter_p1 = iter(train_loader_p1)
    else:
        train_iter_p1 = None
    
    train_iter_p2 = iter(train_loader_p2)
    
    # ==========================================================================
    # Optimizer
    # ==========================================================================
    optimizer = optim.AdamW(model.parameters(), lr=config.lr, weight_decay=0.01)
    
    # Learning rate scheduler: linear warmup then cosine decay
    warmup_steps = min(1000, config.num_steps // 10)
    
    def lr_lambda(step):
        if step < warmup_steps:
            return step / warmup_steps
        else:
            progress = (step - warmup_steps) / (config.num_steps - warmup_steps)
            return 0.5 * (1 + torch.cos(torch.tensor(progress * 3.14159)).item())
    
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    # ==========================================================================
    # Training loop
    # ==========================================================================
    print(f"\nStarting Phase 2 training for {config.num_steps} steps...")
    print(f"  Alignment type: {config.alignment_type}")
    print(f"  Loss weights: recon={config.recon_weight}, align={config.align_weight}, contrastive={config.contrastive_weight}")
    print(f"  Phase 1 mix ratio: {config.phase1_mix_ratio}")
    
    model.train()
    pbar = tqdm(range(1, config.num_steps + 1), desc="Phase 2 Training")
    
    # Metrics accumulators
    metrics_accum = {
        "recon_loss": 0.0,
        "vq_loss": 0.0,
        "align_loss": 0.0,
        "contrastive_loss": 0.0,
        "total_loss": 0.0,
        "code_match_rate": 0.0,
        "cos_sim": 0.0,
        "retrieval_acc": 0.0,
    }
    accum_count = 0
    
    for step in pbar:
        optimizer.zero_grad()
        
        # Freeze encoder if configured
        if step <= config.freeze_encoder_steps:
            for param in model.encoder.parameters():
                param.requires_grad = False
            for param in model.token_embed.parameters():
                param.requires_grad = False
        elif step == config.freeze_encoder_steps + 1:
            for param in model.encoder.parameters():
                param.requires_grad = True
            for param in model.token_embed.parameters():
                param.requires_grad = True
            print(f"\nStep {step}: Unfreezing encoder")
        
        total_loss = torch.tensor(0.0, device=device)
        step_metrics = {}
        
        # ======================================================================
        # Phase 2 batch: Query-Passage alignment
        # ======================================================================
        use_phase2 = random.random() > config.phase1_mix_ratio or train_iter_p1 is None
        
        if use_phase2:
            try:
                batch = next(train_iter_p2)
            except StopIteration:
                train_iter_p2 = iter(train_loader_p2)
                batch = next(train_iter_p2)
            
            # Move to device
            query_tokens = batch["query_tokens"].to(device)
            query_kmer_start = batch["query_kmer_start"].to(device)
            passage_tokens = batch["passage_tokens"].to(device)
            passage_kmer_start = batch["passage_kmer_start"].to(device)
            kmer_len = batch["kmer_len"][0].item()
            
            # Forward pass for query
            query_output = model(query_tokens, query_kmer_start, kmer_len)
            query_logits = query_output["logits"]
            query_z_e = query_output["z_e"]
            query_indices = query_output["indices"]
            query_vq_loss = query_output["vq_loss"]
            
            # Forward pass for passage
            passage_output = model(passage_tokens, passage_kmer_start, kmer_len)
            passage_logits = passage_output["logits"]
            passage_z_e = passage_output["z_e"]
            passage_indices = passage_output["indices"]
            passage_vq_loss = passage_output["vq_loss"]
            
            # Reconstruction losses
            B = query_tokens.shape[0]
            query_targets = torch.stack([
                query_tokens[i, query_kmer_start[i]:query_kmer_start[i]+kmer_len]
                for i in range(B)
            ])
            passage_targets = torch.stack([
                passage_tokens[i, passage_kmer_start[i]:passage_kmer_start[i]+kmer_len]
                for i in range(B)
            ])
            
            query_recon_loss = F.cross_entropy(
                query_logits.view(-1, vocab_size),
                query_targets.view(-1),
            )
            passage_recon_loss = F.cross_entropy(
                passage_logits.view(-1, vocab_size),
                passage_targets.view(-1),
            )
            recon_loss = (query_recon_loss + passage_recon_loss) / 2
            vq_loss = (query_vq_loss + passage_vq_loss) / 2
            
            # Alignment loss
            align_loss, align_metrics = compute_code_alignment_loss(
                query_indices, passage_indices,
                query_z_e, passage_z_e,
                alignment_type=config.alignment_type,
                temperature=config.temperature,
            )
            
            # Contrastive loss
            contrastive_loss = compute_in_batch_contrastive_loss(
                query_z_e, passage_z_e,
                temperature=config.temperature,
            )
            
            # Total loss
            total_loss = (
                config.recon_weight * recon_loss +
                config.commitment_weight * vq_loss +
                config.align_weight * align_loss +
                config.contrastive_weight * contrastive_loss
            )
            
            step_metrics = {
                "recon_loss": recon_loss.item(),
                "vq_loss": vq_loss.item(),
                "align_loss": align_loss.item(),
                "contrastive_loss": contrastive_loss.item(),
                "code_match_rate": align_metrics["code_match_rate"],
                "cos_sim": align_metrics["mean_cos_sim"],
            }
            if "retrieval_acc_q2p" in align_metrics:
                step_metrics["retrieval_acc"] = (align_metrics["retrieval_acc_q2p"] + align_metrics["retrieval_acc_p2q"]) / 2
        
        # ======================================================================
        # Phase 1 batch: Reconstruction only (prevent forgetting)
        # ======================================================================
        else:
            try:
                batch = next(train_iter_p1)
            except StopIteration:
                train_iter_p1 = iter(train_loader_p1)
                batch = next(train_iter_p1)
            
            tokens = batch["tokens"].to(device)
            kmer_start = batch["kmer_start"].to(device)
            kmer_len = batch["kmer_len"][0].item()
            
            output = model(tokens, kmer_start, kmer_len)
            logits = output["logits"]
            vq_loss = output["vq_loss"]
            
            B = tokens.shape[0]
            targets = torch.stack([
                tokens[i, kmer_start[i]:kmer_start[i]+kmer_len]
                for i in range(B)
            ])
            
            recon_loss = F.cross_entropy(
                logits.view(-1, vocab_size),
                targets.view(-1),
            )
            
            total_loss = config.recon_weight * recon_loss + config.commitment_weight * vq_loss
            
            step_metrics = {
                "recon_loss": recon_loss.item(),
                "vq_loss": vq_loss.item(),
                "align_loss": 0.0,
                "contrastive_loss": 0.0,
                "code_match_rate": 0.0,
                "cos_sim": 0.0,
                "retrieval_acc": 0.0,
            }
        
        # Backward pass
        total_loss.backward()
        
        # Gradient clipping
        if config.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
        
        optimizer.step()
        scheduler.step()
        
        # Accumulate metrics
        step_metrics["total_loss"] = total_loss.item()
        for k, v in step_metrics.items():
            metrics_accum[k] += v
        accum_count += 1
        
        # Update progress bar
        pbar.set_postfix({
            "loss": f"{total_loss.item():.4f}",
            "match": f"{step_metrics.get('code_match_rate', 0):.2%}",
            "cos": f"{step_metrics.get('cos_sim', 0):.3f}",
        })
        
        # ======================================================================
        # Logging
        # ======================================================================
        if step % config.log_every == 0:
            avg_metrics = {k: v / accum_count for k, v in metrics_accum.items()}
            avg_metrics["learning_rate"] = scheduler.get_last_lr()[0]
            avg_metrics["step"] = step
            
            wandb.log(avg_metrics)
            
            # Reset accumulators
            for k in metrics_accum:
                metrics_accum[k] = 0.0
            accum_count = 0
        
        # ======================================================================
        # Evaluation
        # ======================================================================
        if step % config.eval_every == 0:
            print(f"\n\nStep {step}: Running evaluation...")
            
            val_metrics = evaluate_phase2(
                model, val_loader_p2, device, config, config.eval_steps
            )
            
            print(f"  Val code match rate: {val_metrics['val_code_match_rate']:.2%}")
            print(f"  Val cosine similarity: {val_metrics['val_cos_sim']:.4f}")
            print(f"  Val retrieval accuracy: {val_metrics['val_retrieval_acc']:.2%}")
            
            val_metrics["step"] = step
            wandb.log(val_metrics)
            
            # Save checkpoint
            torch.save({
                "step": step,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "config": config.to_dict(),
                "phase1_config": phase1_config,
                "val_metrics": val_metrics,
            }, config.save_path)
            print(f"  Saved checkpoint to {config.save_path}")
    
    # ==========================================================================
    # Final save
    # ==========================================================================
    print("\n" + "="*60)
    print("Phase 2 training complete!")
    print("="*60)
    
    torch.save({
        "step": config.num_steps,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "config": config.to_dict(),
        "phase1_config": phase1_config,
    }, config.save_path)
    print(f"Final model saved to {config.save_path}")
    
    wandb.finish()


# Need to import random for phase1_mix_ratio check
import random


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phase 2: Query-Retrieval Alignment")
    parser.add_argument("--phase1_checkpoint", type=str, default="kmer_vqvae_phase1.pt",
                        help="Path to Phase 1 checkpoint")
    parser.add_argument("--num_steps", type=int, default=30000,
                        help="Number of training steps")
    parser.add_argument("--batch_size", type=int, default=128,
                        help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4,
                        help="Learning rate")
    parser.add_argument("--align_weight", type=float, default=0.5,
                        help="Alignment loss weight")
    parser.add_argument("--contrastive_weight", type=float, default=0.3,
                        help="Contrastive loss weight")
    parser.add_argument("--alignment_type", type=str, default="soft",
                        choices=["hard", "soft"],
                        help="Alignment type")
    parser.add_argument("--save_path", type=str, default="kmer_vqvae_phase2.pt",
                        help="Path to save checkpoint")
    parser.add_argument("--wandb_run_name", type=str, default=None,
                        help="Wandb run name")
    
    args = parser.parse_args()
    
    config = Phase2Config(
        phase1_checkpoint=args.phase1_checkpoint,
        num_steps=args.num_steps,
        batch_size=args.batch_size,
        lr=args.lr,
        align_weight=args.align_weight,
        contrastive_weight=args.contrastive_weight,
        alignment_type=args.alignment_type,
        save_path=args.save_path,
        wandb_run_name=args.wandb_run_name or f"phase2-{args.alignment_type}",
    )
    
    train_phase2(config)
