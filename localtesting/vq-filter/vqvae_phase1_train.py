import torch
import torch.nn.functional as F
import torch.optim as optim
from dataclasses import dataclass, field, asdict
from transformers import AutoTokenizer
from tqdm import tqdm
import wandb

from vqvae_phase1_data import make_dataloader
from vqvae_phase1_model import KmerVQVAE


@dataclass
class VQVAEConfig:
    """Configuration for VQ-VAE training."""
    
    # Training
    num_steps: int = 100_000
    batch_size: int = 256
    lr: float = 3e-4
    alpha: float = 0.25  # VQ loss weight in total loss
    grad_clip: float = 1.0
    
    # Model architecture
    d_model: int = 384
    n_layers: int = 4
    n_heads: int = 4
    num_codes: int = 2048
    
    # VQ-specific hyperparameters
    commitment_cost: float = 0.25  # Weight for commitment loss (encoder output → codebook)
    diversity_weight: float = 0.01  # Weight for diversity/entropy regularization
    ema_decay: float = 0.99  # EMA decay rate for codebook updates
    
    # Data
    tokenizer_name: str = "gpt2"
    window_size: int = 64
    kmer_len: int = 4
    stride: int = 32
    max_tokens_total: int = 100_000_000
    
    # Warm start
    warmstart_samples: int = 50_000
    warmstart_iters: int = 20
    
    # Logging & evaluation
    log_every: int = 200
    eval_every: int = 1000
    eval_steps: int = 50
    
    # Paths & wandb
    save_path: str = "kmer_vqvae_phase1.pt"
    wandb_project: str = "kmer-vqvae"
    wandb_run_name: str = None
    
    # Device
    device: str = "cuda"
    
    def to_dict(self) -> dict:
        """Convert config to dictionary for wandb logging."""
        return asdict(self)


@torch.no_grad()
def evaluate(model, val_iter, val_dataloader, device, vocab_size, pad_token_id, eval_steps, alpha, window_size, kmer_len, stride, max_tokens_total, tokenizer_name):
    """Run evaluation on validation set."""
    model.eval()
    
    total_recon_loss = 0.0
    total_vq_loss = 0.0
    total_perplexity = 0.0
    total_diversity_loss = 0.0
    count = 0
    
    for _ in range(eval_steps):
        try:
            batch = next(val_iter)
        except StopIteration:
            # Recreate val iterator
            val_dataloader = make_dataloader(
                batch_size=val_dataloader.batch_size,
                num_workers=2,
                tokenizer_name=tokenizer_name,
                window_size=window_size,
                kmer_len=kmer_len,
                stride=stride * 2,
                max_tokens_total=max_tokens_total // 10,
                verbose=False,
                split="val",
            )
            val_iter = iter(val_dataloader)
            batch = next(val_iter)
        
        tokens = batch["tokens"].to(device)
        kmer_start = batch["kmer_start"].to(device)
        batch_kmer_len = batch["kmer_len"][0].item()
        
        logits, vq_loss, perplexity, _, diversity_loss = model(tokens, kmer_start, batch_kmer_len)
        
        kmer_offsets = torch.arange(batch_kmer_len, device=device)
        kmer_positions = kmer_start.unsqueeze(1) + kmer_offsets.unsqueeze(0)
        target_kmers = torch.gather(tokens, dim=1, index=kmer_positions)
        logits_flat = logits.view(-1, vocab_size)
        targets_flat = target_kmers.view(-1)
        recon_loss = F.cross_entropy(logits_flat, targets_flat, ignore_index=pad_token_id)
        
        total_recon_loss += recon_loss.item()
        total_vq_loss += vq_loss.item()
        total_perplexity += perplexity.item()
        total_diversity_loss += diversity_loss.item()
        count += 1
    
    model.train()
    
    return {
        "val/recon_loss": total_recon_loss / count,
        "val/vq_loss": total_vq_loss / count,
        "val/diversity_loss": total_diversity_loss / count,
        "val/total_loss": (total_recon_loss + alpha * total_vq_loss) / count,
        "val/perplexity": total_perplexity / count,
    }, val_iter, val_dataloader


def train_vqvae(config: VQVAEConfig = None, **kwargs):
    """
    Train a VQ-VAE model on k-mer reconstruction.
    
    Args:
        config: VQVAEConfig object with all hyperparameters.
        **kwargs: Override specific config values.
    """
    # Create config if not provided, and apply any overrides
    if config is None:
        config = VQVAEConfig(**kwargs)
    else:
        # Apply any kwargs overrides to the config
        for key, value in kwargs.items():
            if hasattr(config, key):
                setattr(config, key, value)
    
    # Extract config values for convenience
    num_steps = config.num_steps
    batch_size = config.batch_size
    lr = config.lr
    device = config.device
    log_every = config.log_every
    eval_every = config.eval_every
    eval_steps = config.eval_steps
    save_path = config.save_path
    tokenizer_name = config.tokenizer_name
    alpha = config.alpha
    grad_clip = config.grad_clip
    window_size = config.window_size
    kmer_len = config.kmer_len
    stride = config.stride
    max_tokens_total = config.max_tokens_total
    d_model = config.d_model
    n_layers = config.n_layers
    n_heads = config.n_heads
    num_codes = config.num_codes
    wandb_project = config.wandb_project
    wandb_run_name = config.wandb_run_name
    warmstart_samples = config.warmstart_samples
    warmstart_iters = config.warmstart_iters
    commitment_cost = config.commitment_cost
    diversity_weight = config.diversity_weight
    ema_decay = config.ema_decay
    
    # Initialize wandb with config
    wandb.init(
        project=wandb_project,
        name=wandb_run_name,
        config=config.to_dict(),
    )
    
    # Define step as the x-axis for all metrics
    wandb.define_metric("step")
    wandb.define_metric("train/*", step_metric="step")
    wandb.define_metric("val/*", step_metric="step")
    wandb.define_metric("init/*", step_metric="step")
    
    print(f"Training KmerVQVAE on {device}")
    print(f"  num_steps: {num_steps}")
    print(f"  batch_size: {batch_size}")
    print(f"  lr: {lr}")
    print(f"  alpha (vq_loss weight): {alpha}")
    print(f"  window_size: {window_size}")
    print(f"  kmer_len: {kmer_len}")
    print()
    
    dataloader = make_dataloader(
        batch_size=batch_size,
        num_workers=4,
        tokenizer_name=tokenizer_name,
        window_size=window_size,
        kmer_len=kmer_len,
        stride=stride,
        max_tokens_total=max_tokens_total,
        split="train",
    )
    
    # Create validation dataloader (separate data split)
    print("Creating validation dataloader...")
    val_dataloader = make_dataloader(
        batch_size=batch_size,
        num_workers=2,
        tokenizer_name=tokenizer_name,
        window_size=window_size,
        kmer_len=kmer_len,
        stride=stride * 2,  # Larger stride for less overlap
        max_tokens_total=max_tokens_total // 10,  # Limit val tokens
        verbose=False,
        split="val",
    )
    val_iter = iter(val_dataloader)
    
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    vocab_size = tokenizer.vocab_size
    pad_token_id = tokenizer.pad_token_id
    
    print(f"Vocab size: {vocab_size}")
    print(f"Pad token ID: {pad_token_id}")
    
    model = KmerVQVAE(
        vocab_size=vocab_size,
        d_model=d_model,
        n_layers=n_layers,
        n_heads=n_heads,
        max_len=window_size,
        num_codes=num_codes,
        code_dim=d_model,  # Must equal d_model
        ema_decay=ema_decay,
        commitment_cost=commitment_cost,
        diversity_weight=diversity_weight,
    )
    model = model.to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print()
    
    # Log model info to wandb
    wandb.config.update({
        "total_params": total_params,
        "trainable_params": trainable_params,
        "vocab_size": vocab_size,
    })
    
    # Warm start codebook with k-means
    if warmstart_samples > 0:
        print(f"Warm-starting codebook with k-means ({warmstart_samples} samples, {warmstart_iters} iters)...")
        
        # Create a temporary dataloader for warm start (use train split)
        warmstart_loader = make_dataloader(
            batch_size=batch_size,
            num_workers=4,
            tokenizer_name=tokenizer_name,
            window_size=window_size,
            kmer_len=kmer_len,
            stride=stride,
            max_tokens_total=max_tokens_total,
            verbose=False,
            split="train",
        )
        
        model.warm_start_codebook(
            warmstart_loader,
            kmer_len=kmer_len,
            max_samples=warmstart_samples,
            n_iters=warmstart_iters,
            verbose=True,
        )
        
        # Log initial codebook utilization
        used_codes = (model.vq.cluster_size > 0).sum().item()
        print(f"Initial codebook utilization: {used_codes}/{num_codes} codes")
        wandb.log({"init/codebook_utilization": used_codes, "step": 0})
        print()

    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    
    model.train()
    data_iter = iter(dataloader)
    
    # Running averages for logging
    running_recon_loss = 0.0
    running_vq_loss = 0.0
    running_perplexity = 0.0
    running_diversity_loss = 0.0
    running_count = 0
    
    # Progress bar
    pbar = tqdm(total=num_steps, desc="Training", unit="step")
    
    step = 0
    while step < num_steps:
        # Get next batch
        try:
            batch = next(data_iter)
        except StopIteration:
            print("Recreating data iterator...")
            dataloader = make_dataloader(
                batch_size=batch_size,
                num_workers=4,
                tokenizer_name=tokenizer_name,
                window_size=window_size,
                kmer_len=kmer_len,
                stride=stride,
                max_tokens_total=max_tokens_total,
                verbose=False,
                split="train",
            )
            data_iter = iter(dataloader)
            batch = next(data_iter)
        
        tokens = batch["tokens"].to(device)
        kmer_start = batch["kmer_start"].to(device)
        batch_kmer_len = batch["kmer_len"][0].item()
        
        # Forward pass
        logits, vq_loss, perplexity, _, diversity_loss = model(tokens, kmer_start, batch_kmer_len)
        
        B = tokens.shape[0]
        kmer_offsets = torch.arange(batch_kmer_len, device=device)
        kmer_positions = kmer_start.unsqueeze(1) + kmer_offsets.unsqueeze(0) 
        target_kmers = torch.gather(tokens, dim=1, index=kmer_positions)        
        logits_flat = logits.view(-1, vocab_size)
        targets_flat = target_kmers.view(-1)
        recon_loss = F.cross_entropy(logits_flat, targets_flat, ignore_index=pad_token_id)
        total_loss = recon_loss + alpha * vq_loss
        
        optimizer.zero_grad()
        total_loss.backward()
        
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        
        optimizer.step()
        
        running_recon_loss += recon_loss.item()
        running_vq_loss += vq_loss.item()
        running_perplexity += perplexity.item()
        running_diversity_loss += diversity_loss.item()
        running_count += 1
        
        step += 1
        pbar.update(1)
        
        # Log metrics
        if step % log_every == 0:
            avg_recon = running_recon_loss / running_count
            avg_vq = running_vq_loss / running_count
            avg_perp = running_perplexity / running_count
            avg_div = running_diversity_loss / running_count
            
            # Update progress bar description
            pbar.set_postfix({
                "recon": f"{avg_recon:.4f}",
                "vq": f"{avg_vq:.4f}",
                "div": f"{avg_div:.4f}",
                "perp": f"{avg_perp:.1f}",
            })
            
            # Log to wandb
            wandb.log({
                "train/recon_loss": avg_recon,
                "train/vq_loss": avg_vq,
                "train/diversity_loss": avg_div,
                "train/total_loss": avg_recon + alpha * avg_vq,
                "train/perplexity": avg_perp,
                "step": step,
            })
            
            # Reset running averages
            running_recon_loss = 0.0
            running_vq_loss = 0.0
            running_perplexity = 0.0
            running_diversity_loss = 0.0
            running_count = 0
        
        # Evaluation
        if step % eval_every == 0:
            val_metrics, val_iter, val_dataloader = evaluate(
                model, val_iter, val_dataloader, device, vocab_size, pad_token_id,
                eval_steps, alpha, window_size, kmer_len, stride, max_tokens_total, tokenizer_name
            )
            # Log val metrics with same step (won't create duplicate x-axis points)
            wandb.log({**val_metrics, "step": step})
            tqdm.write(
                f"  [Val @ step {step}] recon: {val_metrics['val/recon_loss']:.4f} | "
                f"vq: {val_metrics['val/vq_loss']:.4f} | "
                f"perp: {val_metrics['val/perplexity']:.1f}"
            )
    
    pbar.close()
    
    # Save model
    print(f"\nTraining complete. Saving model to {save_path}")
    
    save_dict = {
        "model_state_dict": model.state_dict(),
        "tokenizer_name": tokenizer_name,
        "config": {
            "vocab_size": vocab_size,
            "d_model": d_model,
            "n_layers": n_layers,
            "n_heads": n_heads,
            "max_len": window_size,
            "num_codes": num_codes,
            "code_dim": d_model,
        },
    }
    
    torch.save(save_dict, save_path)
    print(f"Model saved successfully!")
    
    # Save model artifact to wandb
    artifact = wandb.Artifact(
        name=f"kmer-vqvae-{wandb.run.id}",
        type="model",
        description="Trained KmerVQVAE model checkpoint",
    )
    artifact.add_file(save_path)
    wandb.log_artifact(artifact)
    
    # Finish wandb run
    wandb.finish()
    
    return model


if __name__ == "__main__":
    # Create config with desired hyperparameters
    config = VQVAEConfig(
        # Training
        num_steps=50_000,
        batch_size=256,
        lr=2e-4,
        alpha=1.0,
        grad_clip=1.0,
        
        # Model
        d_model=384,
        n_layers=4,
        n_heads=4,
        num_codes=2048,

        # VQ
        commitment_cost= 0.25,
        diversity_weight= 0.001,
        ema_decay= 0.993,
        
        # Data
        tokenizer_name="gpt2",
        window_size=64,
        kmer_len=4,
        stride=32,
        max_tokens_total=100_000_000,
        
        # Warm start
        warmstart_samples=50_000,
        warmstart_iters=20,
        
        # Logging
        log_every=100,
        eval_every=1000,
        eval_steps=50,
        
        # Paths
        save_path="kmer_vqvae_phase1.pt",
        wandb_project="kmer-vqvae",
        wandb_run_name=None,
        
        # Device
        device="cuda" if torch.cuda.is_available() else "cpu",
    )
    
    # Train with config
    train_vqvae(config)
