"""
Phase 1 Training with MiniPile (Single-Stage Alternative)

This is an alternative to the curriculum learning approach. MiniPile already
contains diverse domains (academic, code, web, conversational), so we can
potentially train a well-rounded codebook in a single phase.

Usage:
    python vqvae_phase1_train_minipile.py
    
    # With custom settings
    python vqvae_phase1_train_minipile.py --num_codes 1024 --max_texts 500000
"""

import argparse
import torch
import torch.optim as optim
from dataclasses import dataclass, asdict
from transformers import AutoTokenizer
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import wandb

from vqvae_phase1_model import KmerVQVAE
from vqvae_phase1_data_minipile import make_minipile_dataloader


@dataclass
class MiniPileTrainConfig:
    """Configuration for MiniPile training."""
    
    # Model architecture
    vocab_size: int = 50257  # GPT-2 vocab
    d_model: int = 384
    n_layers: int = 4
    n_heads: int = 4
    num_codes: int = 1024
    
    # VQ hyperparameters
    commitment_cost: float = 0.25
    diversity_weight: float = 0.2  # Anti-collapse
    ema_decay: float = 0.99
    dead_code_threshold: float = 1.0
    dead_code_restart_prob: float = 0.01
    
    # Data
    tokenizer_name: str = "gpt2"
    window_size: int = 64
    kmer_len: int = 4
    max_texts: int = 1_000_000  # All of MiniPile
    streaming: bool = False  # Set True for memory efficiency
    
    # Training
    batch_size: int = 256
    lr: float = 1e-4
    num_steps: int = 100_000
    grad_clip: float = 1.0
    
    # Teacher for warm start
    teacher_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    use_warm_start: bool = True
    warm_start_samples: int = 10_000
    
    # Logging
    log_every: int = 100
    eval_every: int = 2000
    save_path: str = "kmer_vqvae_minipile.pt"
    wandb_project: str = "kmer-vqvae"
    wandb_run_name: str = "minipile-phase1"
    
    # Device
    device: str = "cuda"
    
    def to_dict(self):
        return asdict(self)


def train_minipile(config: MiniPileTrainConfig):
    """Main training loop for MiniPile."""
    
    device = torch.device(config.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # ==========================================================================
    # Initialize model
    # ==========================================================================
    model = KmerVQVAE(
        vocab_size=config.vocab_size,
        d_model=config.d_model,
        n_layers=config.n_layers,
        n_heads=config.n_heads,
        num_codes=config.num_codes,
        commitment_cost=config.commitment_cost,
        diversity_weight=config.diversity_weight,
        ema_decay=config.ema_decay,
        dead_code_threshold=config.dead_code_threshold,
        dead_code_restart_prob=config.dead_code_restart_prob,
    ).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {total_params:,} total, {trainable_params:,} trainable")
    
    # ==========================================================================
    # Warm start codebook with teacher
    # ==========================================================================
    if config.use_warm_start:
        print(f"\nWarm starting codebook with {config.teacher_model}...")
        
        teacher = SentenceTransformer(config.teacher_model)
        tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # Get sample texts from MiniPile for warm start
        from vqvae_phase1_data_minipile import load_minipile
        sample_texts = load_minipile(
            split="train",
            max_texts=config.warm_start_samples,
            verbose=True,
        )
        
        # Warm start
        model.warm_start_codebook_with_teacher(
            teacher_model=teacher,
            sample_texts=sample_texts,
            num_samples=config.warm_start_samples,
            device=device,
        )
        
        del teacher
        torch.cuda.empty_cache()
    
    # ==========================================================================
    # Setup data
    # ==========================================================================
    print("\nSetting up MiniPile dataloader...")
    dataloader = make_minipile_dataloader(
        batch_size=config.batch_size,
        num_workers=4,
        tokenizer_name=config.tokenizer_name,
        window_size=config.window_size,
        kmer_len=config.kmer_len,
        max_texts=config.max_texts,
        streaming=config.streaming,
        verbose=True,
    )
    
    data_iter = iter(dataloader)
    
    # ==========================================================================
    # Optimizer and scheduler
    # ==========================================================================
    optimizer = optim.AdamW(model.parameters(), lr=config.lr, weight_decay=0.01)
    
    # Cosine schedule with warmup
    warmup_steps = min(2000, config.num_steps // 10)
    
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
    print(f"\nStarting MiniPile training for {config.num_steps} steps...")
    print(f"  Codebook size: {config.num_codes}")
    print(f"  Diversity weight: {config.diversity_weight}")
    print(f"  Batch size: {config.batch_size}")
    
    model.train()
    pbar = tqdm(range(1, config.num_steps + 1), desc="Training")
    
    # Metrics accumulators
    metrics_accum = {
        "recon_loss": 0.0,
        "vq_loss": 0.0,
        "diversity_loss": 0.0,
        "total_loss": 0.0,
        "perplexity": 0.0,
        "dead_codes": 0.0,
    }
    accum_count = 0
    
    for step in pbar:
        optimizer.zero_grad()
        
        # Get batch
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            batch = next(data_iter)
        
        tokens = batch["tokens"].to(device)
        kmer_start = batch["kmer_start"].to(device)
        kmer_len = batch["kmer_len"][0].item()
        
        # Forward pass
        output = model(tokens, kmer_start, kmer_len)
        
        # Reconstruction loss
        logits = output["logits"]  # [B, kmer_len, vocab]
        B = tokens.shape[0]
        
        # Extract target k-mers
        targets = []
        for i in range(B):
            start = kmer_start[i].item()
            targets.append(tokens[i, start : start + kmer_len])
        targets = torch.stack(targets)  # [B, kmer_len]
        
        recon_loss = torch.nn.functional.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            targets.reshape(-1),
        )
        
        # Total loss
        total_loss = recon_loss + output["vq_loss"]
        
        # Backward
        total_loss.backward()
        
        if config.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
        
        optimizer.step()
        scheduler.step()
        
        # Accumulate metrics
        metrics_accum["recon_loss"] += recon_loss.item()
        metrics_accum["vq_loss"] += output["vq_loss"].item()
        metrics_accum["diversity_loss"] += output.get("diversity_loss", torch.tensor(0.0)).item()
        metrics_accum["total_loss"] += total_loss.item()
        metrics_accum["perplexity"] += output["perplexity"].item()
        
        # Count dead codes
        with torch.no_grad():
            usage = model.vq.cluster_usage
            dead = (usage < config.dead_code_threshold).sum().item()
            metrics_accum["dead_codes"] += dead
        
        accum_count += 1
        
        pbar.set_postfix({
            "loss": f"{total_loss.item():.4f}",
            "ppl": f"{output['perplexity'].item():.1f}",
            "dead": dead,
        })
        
        # Logging
        if step % config.log_every == 0:
            avg_metrics = {k: v / accum_count for k, v in metrics_accum.items()}
            avg_metrics["learning_rate"] = scheduler.get_last_lr()[0]
            avg_metrics["step"] = step
            avg_metrics["codebook_utilization"] = (config.num_codes - avg_metrics["dead_codes"]) / config.num_codes
            
            wandb.log(avg_metrics)
            
            # Log codebook usage histogram
            if step % (config.log_every * 10) == 0:
                with torch.no_grad():
                    usage = model.vq.cluster_usage.cpu().numpy()
                    wandb.log({"codebook_usage": wandb.Histogram(usage), "step": step})
            
            for k in metrics_accum:
                metrics_accum[k] = 0.0
            accum_count = 0
        
        # Save checkpoint
        if step % config.eval_every == 0:
            print(f"\n\nStep {step}: Saving checkpoint...")
            print(f"  Perplexity: {output['perplexity'].item():.1f}")
            print(f"  Dead codes: {dead}/{config.num_codes}")
            print(f"  Utilization: {(config.num_codes - dead) / config.num_codes:.1%}")
            
            torch.save({
                "step": step,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "config": config.to_dict(),
                "perplexity": output["perplexity"].item(),
            }, config.save_path)
            print(f"  Saved to {config.save_path}")
    
    # Final save
    print("\n" + "="*60)
    print("MiniPile training complete!")
    print("="*60)
    
    torch.save({
        "step": config.num_steps,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "config": config.to_dict(),
        "perplexity": output["perplexity"].item(),
    }, config.save_path)
    print(f"Final model saved to {config.save_path}")
    
    wandb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train VQ-VAE on MiniPile")
    parser.add_argument("--num_codes", type=int, default=1024)
    parser.add_argument("--num_steps", type=int, default=100000)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--max_texts", type=int, default=1000000)
    parser.add_argument("--diversity_weight", type=float, default=0.2)
    parser.add_argument("--streaming", action="store_true")
    parser.add_argument("--save_path", type=str, default="kmer_vqvae_minipile.pt")
    parser.add_argument("--no_warm_start", action="store_true")
    
    args = parser.parse_args()
    
    config = MiniPileTrainConfig(
        num_codes=args.num_codes,
        num_steps=args.num_steps,
        batch_size=args.batch_size,
        lr=args.lr,
        max_texts=args.max_texts,
        diversity_weight=args.diversity_weight,
        streaming=args.streaming,
        save_path=args.save_path,
        use_warm_start=not args.no_warm_start,
        wandb_run_name=f"minipile-k{args.num_codes}-div{args.diversity_weight}",
    )
    
    train_minipile(config)
