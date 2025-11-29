import torch
import torch.nn.functional as F
import torch.optim as optim
from transformers import AutoTokenizer
from tqdm import tqdm
import wandb

from vqvae_phase1_data import make_dataloader
from vqvae_phase1_model import KmerVQVAE


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


def train_vqvae(
    num_steps: int = 100_000,
    batch_size: int = 256,
    lr: float = 3e-4,
    device: str = "cuda",
    log_every: int = 200,
    eval_every: int = 1000,
    eval_steps: int = 50,
    save_path: str = "kmer_vqvae_phase1.pt",
    tokenizer_name: str = "gpt2",
    alpha: float = 0.25,
    grad_clip: float = 1.0,
    window_size: int = 256,
    kmer_len: int = 16,
    stride: int = 128,
    max_tokens_total: int = 100_000_000,
    d_model: int = 256,
    n_layers: int = 4,
    n_heads: int = 4,
    num_codes: int = 2048,
    wandb_project: str = "kmer-vqvae",
    wandb_run_name: str = None,
):
    # Initialize wandb
    config = {
        "num_steps": num_steps,
        "batch_size": batch_size,
        "lr": lr,
        "alpha": alpha,
        "grad_clip": grad_clip,
        "window_size": window_size,
        "kmer_len": kmer_len,
        "stride": stride,
        "max_tokens_total": max_tokens_total,
        "d_model": d_model,
        "n_layers": n_layers,
        "n_heads": n_heads,
        "num_codes": num_codes,
        "tokenizer_name": tokenizer_name,
        "eval_every": eval_every,
        "eval_steps": eval_steps,
    }
    
    wandb.init(
        project=wandb_project,
        name=wandb_run_name,
        config=config,
    )
    
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
    )
    
    # Create validation dataloader (smaller, separate data)
    print("Creating validation dataloader...")
    val_dataloader = make_dataloader(
        batch_size=batch_size,
        num_workers=2,
        tokenizer_name=tokenizer_name,
        window_size=window_size,
        kmer_len=kmer_len,
        stride=stride * 2,  # Larger stride for less overlap
        max_tokens_total=max_tokens_total // 10,  # 10% of training data
        verbose=False,
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
            wandb.log({**val_metrics, "step": step})
            tqdm.write(
                f"  [Val] recon: {val_metrics['val/recon_loss']:.4f} | "
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
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    train_vqvae(
        num_steps=50_000,
        batch_size=256,
        lr=2e-4,
        device=device,
        log_every=100,
        save_path="kmer_vqvae_phase1.pt",
        tokenizer_name="gpt2",
    )
