"""
Phase 1b Training: Domain Fine-tuning with LoRA

Fine-tunes a pre-trained VQ-VAE (from Phase 1a web text) on academic and
conversational domains using LoRA adapters for efficient training.

The base model and codebook are frozen; only LoRA parameters are trained.
Optionally, the codebook can be lightly fine-tuned with a lower learning rate.

Usage:
    python vqvae_phase1b_train_lora.py --checkpoint kmer_vqvae_phase1.pt --domain both
"""

import argparse
import os
import torch
import torch.nn.functional as F
import torch.optim as optim
from dataclasses import dataclass, asdict
from transformers import AutoTokenizer
from tqdm import tqdm
import wandb

from vqvae_phase1b_data_domains import make_dataloader
from vqvae_phase1_model import KmerVQVAE
from lora_adapters import (
    add_lora_to_model,
    get_lora_parameters,
    freeze_base_model,
    count_parameters,
    ProjectionAdapter,
)


@dataclass
class Phase1bConfig:
    """Configuration for Phase 1b domain fine-tuning with LoRA."""
    
    # Phase 1a checkpoint (required)
    phase1a_checkpoint: str = "kmer_vqvae_phase1.pt"
    
    # Domain to fine-tune on
    domain: str = "both"  # "academic", "conversational", or "both"
    
    # Training
    num_steps: int = 20_000
    batch_size: int = 256
    lr: float = 1e-4  # Learning rate for LoRA parameters
    codebook_lr: float = 1e-5  # Lower LR for optional codebook fine-tuning
    grad_clip: float = 1.0
    
    # LoRA settings
    lora_rank: int = 8
    lora_alpha: float = 16.0
    lora_dropout: float = 0.05
    lora_target_modules: str = "q_proj,v_proj"  # Comma-separated
    
    # What to train
    train_codebook: bool = False  # Whether to fine-tune codebook (with lower LR)
    train_decoder: bool = True   # Whether to fine-tune decoder
    
    # Loss weights
    recon_weight: float = 1.0
    vq_weight: float = 0.25
    diversity_weight: float = 0.1  # Lower than Phase 1a
    
    # Data
    tokenizer_name: str = "gpt2"
    window_size: int = 64
    kmer_len: int = 4
    max_tokens_total: int = 100_000_000
    
    # Logging
    log_every: int = 100
    eval_every: int = 1000
    eval_steps: int = 50
    
    # Paths
    save_path: str = "kmer_vqvae_phase1b_lora.pt"
    wandb_project: str = "kmer-vqvae"
    wandb_run_name: str = "phase1b-lora"
    
    # Device
    device: str = "cuda"
    
    def to_dict(self):
        return asdict(self)


@torch.no_grad()
def evaluate(model, val_loader, device, vocab_size, eval_steps, config):
    """Evaluate model on validation set."""
    model.eval()
    
    total_recon_loss = 0.0
    total_vq_loss = 0.0
    total_perplexity = 0.0
    count = 0
    
    val_iter = iter(val_loader)
    
    for _ in range(eval_steps):
        try:
            batch = next(val_iter)
        except StopIteration:
            val_iter = iter(val_loader)
            batch = next(val_iter)
        
        tokens = batch["tokens"].to(device)
        kmer_start = batch["kmer_start"].to(device)
        kmer_len = batch["kmer_len"][0].item()
        
        output = model(tokens, kmer_start, kmer_len)
        logits = output["logits"]
        vq_loss = output["vq_loss"]
        perplexity = output["perplexity"]
        
        B = tokens.shape[0]
        targets = torch.stack([
            tokens[i, kmer_start[i]:kmer_start[i]+kmer_len]
            for i in range(B)
        ])
        
        recon_loss = F.cross_entropy(
            logits.view(-1, vocab_size),
            targets.view(-1),
        )
        
        total_recon_loss += recon_loss.item()
        total_vq_loss += vq_loss.item()
        total_perplexity += perplexity.item()
        count += 1
    
    model.train()
    
    return {
        "val_recon_loss": total_recon_loss / count,
        "val_vq_loss": total_vq_loss / count,
        "val_perplexity": total_perplexity / count,
    }


def train_phase1b(config: Phase1bConfig):
    """Main Phase 1b training loop with LoRA."""
    
    device = torch.device(config.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # ==========================================================================
    # Load Phase 1a checkpoint
    # ==========================================================================
    print(f"\nLoading Phase 1a checkpoint: {config.phase1a_checkpoint}")
    
    if not os.path.exists(config.phase1a_checkpoint):
        raise FileNotFoundError(f"Checkpoint not found: {config.phase1a_checkpoint}")
    
    checkpoint = torch.load(config.phase1a_checkpoint, map_location=device)
    phase1a_config = checkpoint.get("config", {})
    
    # Create model with Phase 1a architecture
    model = KmerVQVAE(
        vocab_size=phase1a_config.get("vocab_size", 50257),
        d_model=phase1a_config.get("d_model", 384),
        n_layers=phase1a_config.get("n_layers", 4),
        n_heads=phase1a_config.get("n_heads", 4),
        num_codes=phase1a_config.get("num_codes", 1024),
        commitment_cost=config.vq_weight,
        diversity_weight=config.diversity_weight,
        ema_decay=phase1a_config.get("ema_decay", 0.99),
        dead_code_threshold=phase1a_config.get("dead_code_threshold", 1.0),
        dead_code_restart_prob=0.0,  # Disable restarts for fine-tuning
    ).to(device)
    
    model.load_state_dict(checkpoint["model_state_dict"])
    print(f"  Loaded model from step {checkpoint.get('step', 'unknown')}")
    
    # ==========================================================================
    # Add LoRA to encoder
    # ==========================================================================
    print("\nAdding LoRA adapters to encoder...")
    
    target_modules = config.lora_target_modules.split(",")
    model, lora_params = add_lora_to_model(
        model,
        rank=config.lora_rank,
        alpha=config.lora_alpha,
        target_modules=target_modules,
        dropout=config.lora_dropout,
    )
    
    # Freeze base model
    freeze_base_model(model, freeze_codebook=not config.train_codebook)
    
    # Unfreeze decoder if configured
    if config.train_decoder:
        for param in model.decoder.parameters():
            param.requires_grad = True
    
    # Count parameters
    param_counts = count_parameters(model)
    print(f"  Total parameters: {param_counts['total']:,}")
    print(f"  Trainable parameters: {param_counts['trainable']:,} ({param_counts['trainable_pct']:.2f}%)")
    print(f"  LoRA parameters: {param_counts['lora']:,}")
    
    # ==========================================================================
    # Setup tokenizer and data
    # ==========================================================================
    tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    vocab_size = tokenizer.vocab_size
    
    print(f"\nLoading {config.domain} domain data...")
    
    train_loader = make_dataloader(
        domain=config.domain,
        batch_size=config.batch_size,
        num_workers=4,
        tokenizer_name=config.tokenizer_name,
        window_size=config.window_size,
        kmer_len=config.kmer_len,
        max_tokens_total=config.max_tokens_total,
        verbose=True,
        split="train",
    )
    
    val_loader = make_dataloader(
        domain=config.domain,
        batch_size=config.batch_size,
        num_workers=2,
        tokenizer_name=config.tokenizer_name,
        window_size=config.window_size,
        kmer_len=config.kmer_len,
        max_tokens_total=config.max_tokens_total // 10,
        verbose=True,
        split="val",
    )
    
    train_iter = iter(train_loader)
    
    # ==========================================================================
    # Optimizer with different LR for LoRA vs codebook
    # ==========================================================================
    param_groups = []
    
    # LoRA parameters
    lora_params_list = get_lora_parameters(model)
    if lora_params_list:
        param_groups.append({
            "params": lora_params_list,
            "lr": config.lr,
            "name": "lora",
        })
    
    # Decoder parameters (if training)
    if config.train_decoder:
        decoder_params = list(model.decoder.parameters())
        param_groups.append({
            "params": decoder_params,
            "lr": config.lr,
            "name": "decoder",
        })
    
    # Codebook parameters (if training, with lower LR)
    if config.train_codebook:
        codebook_params = [model.vq.codebook]
        param_groups.append({
            "params": codebook_params,
            "lr": config.codebook_lr,
            "name": "codebook",
        })
    
    optimizer = optim.AdamW(param_groups, weight_decay=0.01)
    
    # Learning rate scheduler
    warmup_steps = min(500, config.num_steps // 10)
    
    def lr_lambda(step):
        if step < warmup_steps:
            return step / warmup_steps
        return 1.0  # Constant LR after warmup for fine-tuning
    
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
    print(f"\nStarting Phase 1b LoRA training for {config.num_steps} steps...")
    print(f"  Domain: {config.domain}")
    print(f"  LoRA rank: {config.lora_rank}, alpha: {config.lora_alpha}")
    print(f"  Training codebook: {config.train_codebook}")
    print(f"  Training decoder: {config.train_decoder}")
    
    model.train()
    pbar = tqdm(range(1, config.num_steps + 1), desc="Phase 1b Training")
    
    # Metrics accumulators
    metrics_accum = {
        "recon_loss": 0.0,
        "vq_loss": 0.0,
        "perplexity": 0.0,
        "total_loss": 0.0,
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
        
        tokens = batch["tokens"].to(device)
        kmer_start = batch["kmer_start"].to(device)
        kmer_len = batch["kmer_len"][0].item()
        
        # Forward pass
        output = model(tokens, kmer_start, kmer_len)
        logits = output["logits"]
        vq_loss = output["vq_loss"]
        perplexity = output["perplexity"]
        diversity_loss = output.get("diversity_loss", 0.0)
        
        # Reconstruction loss
        B = tokens.shape[0]
        targets = torch.stack([
            tokens[i, kmer_start[i]:kmer_start[i]+kmer_len]
            for i in range(B)
        ])
        
        recon_loss = F.cross_entropy(
            logits.view(-1, vocab_size),
            targets.view(-1),
        )
        
        # Total loss
        total_loss = (
            config.recon_weight * recon_loss +
            config.vq_weight * vq_loss
        )
        if isinstance(diversity_loss, torch.Tensor):
            total_loss += config.diversity_weight * diversity_loss
        
        # Backward pass
        total_loss.backward()
        
        if config.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
        
        optimizer.step()
        scheduler.step()
        
        # Accumulate metrics
        metrics_accum["recon_loss"] += recon_loss.item()
        metrics_accum["vq_loss"] += vq_loss.item()
        metrics_accum["perplexity"] += perplexity.item()
        metrics_accum["total_loss"] += total_loss.item()
        accum_count += 1
        
        pbar.set_postfix({
            "loss": f"{total_loss.item():.4f}",
            "ppl": f"{perplexity.item():.1f}",
        })
        
        # Logging
        if step % config.log_every == 0:
            avg_metrics = {k: v / accum_count for k, v in metrics_accum.items()}
            avg_metrics["learning_rate"] = scheduler.get_last_lr()[0]
            avg_metrics["step"] = step
            
            wandb.log(avg_metrics)
            
            # Reset accumulators
            for k in metrics_accum:
                metrics_accum[k] = 0.0
            accum_count = 0
        
        # Evaluation
        if step % config.eval_every == 0:
            print(f"\n\nStep {step}: Running evaluation...")
            
            val_metrics = evaluate(
                model, val_loader, device, vocab_size, config.eval_steps, config
            )
            
            print(f"  Val recon loss: {val_metrics['val_recon_loss']:.4f}")
            print(f"  Val perplexity: {val_metrics['val_perplexity']:.1f}")
            
            val_metrics["step"] = step
            wandb.log(val_metrics)
            
            # Save checkpoint
            save_checkpoint(model, optimizer, step, config, phase1a_config, val_metrics)
    
    # Final save
    print("\n" + "="*60)
    print("Phase 1b training complete!")
    print("="*60)
    
    save_checkpoint(model, optimizer, config.num_steps, config, phase1a_config, {})
    
    wandb.finish()


def save_checkpoint(model, optimizer, step, config, phase1a_config, val_metrics):
    """Save model checkpoint."""
    torch.save({
        "step": step,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "config": config.to_dict(),
        "phase1a_config": phase1a_config,
        "val_metrics": val_metrics,
    }, config.save_path)
    print(f"  Saved checkpoint to {config.save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phase 1b: Domain Fine-tuning with LoRA")
    parser.add_argument("--checkpoint", type=str, default="kmer_vqvae_phase1.pt",
                        help="Path to Phase 1a checkpoint")
    parser.add_argument("--domain", type=str, default="both",
                        choices=["academic", "conversational", "both"],
                        help="Domain to fine-tune on")
    parser.add_argument("--num_steps", type=int, default=20000,
                        help="Number of training steps")
    parser.add_argument("--lora_rank", type=int, default=8,
                        help="LoRA rank")
    parser.add_argument("--lr", type=float, default=1e-4,
                        help="Learning rate for LoRA")
    parser.add_argument("--train_codebook", action="store_true",
                        help="Also fine-tune codebook with lower LR")
    parser.add_argument("--save_path", type=str, default="kmer_vqvae_phase1b_lora.pt",
                        help="Path to save checkpoint")
    
    args = parser.parse_args()
    
    config = Phase1bConfig(
        phase1a_checkpoint=args.checkpoint,
        domain=args.domain,
        num_steps=args.num_steps,
        lora_rank=args.lora_rank,
        lr=args.lr,
        train_codebook=args.train_codebook,
        save_path=args.save_path,
        wandb_run_name=f"phase1b-{args.domain}-lora-r{args.lora_rank}",
    )
    
    train_phase1b(config)
