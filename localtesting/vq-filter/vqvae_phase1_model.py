import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class RotaryPositionEmbedding(nn.Module):
    """
    Rotary Position Embedding (RoPE).
    Applies rotation to query and key vectors based on position.
    """
    
    def __init__(self, dim: int, max_len: int = 512, base: float = 10000.0):
        """
        Args:
            dim: Dimension of the embedding (should be head_dim).
            max_len: Maximum sequence length.
            base: Base for the frequency computation.
        """
        super().__init__()
        self.dim = dim
        self.max_len = max_len
        self.base = base
        
        # Precompute inverse frequencies
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        
        # Precompute cos and sin for all positions
        self._build_cache(max_len)
    
    def _build_cache(self, seq_len: int):
        """Build cos/sin cache for given sequence length."""
        positions = torch.arange(seq_len, device=self.inv_freq.device).float()
        freqs = torch.outer(positions, self.inv_freq)  # [seq_len, dim/2]
        
        # Duplicate frequencies for pairing
        emb = torch.cat([freqs, freqs], dim=-1)  # [seq_len, dim]
        
        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)
    
    def forward(self, x: torch.Tensor, seq_len: int = None):
        """
        Args:
            x: Input tensor [B, n_heads, T, head_dim].
            seq_len: Sequence length (optional, will use x.shape[2] if not provided).
        
        Returns:
            Tuple of (cos, sin) tensors for the given sequence length.
        """
        if seq_len is None:
            seq_len = x.shape[2]
        
        if seq_len > self.cos_cached.shape[0]:
            self._build_cache(seq_len)
        
        return (
            self.cos_cached[:seq_len].to(x.dtype),
            self.sin_cached[:seq_len].to(x.dtype),
        )


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotate half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat([-x2, x1], dim=-1)


def apply_rotary_pos_emb(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
    """
    Apply rotary position embedding to query and key tensors.
    
    Args:
        q: Query tensor [B, n_heads, T, head_dim].
        k: Key tensor [B, n_heads, T, head_dim].
        cos: Cosine tensor [T, head_dim].
        sin: Sine tensor [T, head_dim].
    
    Returns:
        Tuple of (rotated_q, rotated_k).
    """
    # Expand cos/sin for batch and heads: [1, 1, T, head_dim]
    cos = cos.unsqueeze(0).unsqueeze(0)
    sin = sin.unsqueeze(0).unsqueeze(0)
    
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    
    return q_embed, k_embed


class RoPEMultiHeadAttention(nn.Module):
    """
    Multi-head attention with Rotary Position Embeddings.
    """
    
    def __init__(self, d_model: int, n_heads: int, max_len: int = 512, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        
        self.rope = RotaryPositionEmbedding(self.head_dim, max_len)
        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.head_dim)
    
    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            x: Input tensor [B, T, d_model].
            attn_mask: Optional attention mask.
        
        Returns:
            Output tensor [B, T, d_model].
        """
        B, T, _ = x.shape
        
        # Project to Q, K, V
        q = self.q_proj(x)  # [B, T, d_model]
        k = self.k_proj(x)
        v = self.v_proj(x)
        
        # Reshape for multi-head attention: [B, n_heads, T, head_dim]
        q = q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        
        # Apply RoPE to Q and K
        cos, sin = self.rope(q, T)
        q, k = apply_rotary_pos_emb(q, k, cos, sin)
        
        # Scaled dot-product attention
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) / self.scale  # [B, n_heads, T, T]
        
        if attn_mask is not None:
            attn_weights = attn_weights + attn_mask
        
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention to values
        attn_output = torch.matmul(attn_weights, v)  # [B, n_heads, T, head_dim]
        
        # Reshape back: [B, T, d_model]
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, T, self.d_model)
        
        return self.out_proj(attn_output)


class RoPETransformerEncoderLayer(nn.Module):
    """
    Transformer encoder layer with RoPE attention.
    """
    
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        max_len: int = 512,
        dropout: float = 0.1,
        dim_feedforward: int = None,
    ):
        super().__init__()
        if dim_feedforward is None:
            dim_feedforward = d_model * 4
        
        self.self_attn = RoPEMultiHeadAttention(d_model, n_heads, max_len, dropout)
        
        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
            nn.Dropout(dropout),
        )
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor = None) -> torch.Tensor:
        # Self-attention with residual
        attn_out = self.self_attn(x, attn_mask)
        x = self.norm1(x + self.dropout(attn_out))
        
        # FFN with residual
        ffn_out = self.ffn(x)
        x = self.norm2(x + ffn_out)
        
        return x


class TransformerEncoder(nn.Module):
    """
    Transformer encoder for token sequences with RoPE position encoding.
    """
    
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 256,
        n_layers: int = 4,
        n_heads: int = 4,
        max_len: int = 512,
        dropout: float = 0.1,
    ):
        """
        Args:
            vocab_size: Size of vocabulary.
            d_model: Model dimension.
            n_layers: Number of transformer layers.
            n_heads: Number of attention heads.
            max_len: Maximum sequence length.
            dropout: Dropout rate.
        """
        super().__init__()
        self.d_model = d_model
        self.max_len = max_len
        
        # Token embedding only (no position embedding - using RoPE)
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        
        # Transformer encoder layers with RoPE
        self.layers = nn.ModuleList([
            RoPETransformerEncoderLayer(
                d_model=d_model,
                n_heads=n_heads,
                max_len=max_len,
                dropout=dropout,
            )
            for _ in range(n_layers)
        ])
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """
        Args:
            token_ids: [B, T] tensor of token IDs.
        
        Returns:
            Hidden states [B, T, d_model].
        """
        # Get token embeddings (no position embeddings - RoPE handles positions)
        x = self.token_embedding(token_ids)  # [B, T, d_model]
        x = self.dropout(x)
        
        # Pass through transformer encoder layers
        for layer in self.layers:
            x = layer(x)
        
        return x


class VectorQuantizerEMA(nn.Module):
    """
    Vector Quantizer with Exponential Moving Average updates and diversity regularization.
    """
    
    def __init__(
        self,
        num_codes: int,
        code_dim: int,
        decay: float = 0.995,
        eps: float = 1e-8,
        commitment_cost: float = 0.3,
        diversity_weight: float = 0.001,
    ):
        """
        Args:
            num_codes: Number of codes in the codebook.
            code_dim: Dimension of each code vector.
            decay: EMA decay rate.
            eps: Epsilon for numerical stability.
            commitment_cost: Weight for commitment loss.
            diversity_weight: Weight for diversity regularization (entropy maximization).
        """
        super().__init__()
        self.num_codes = num_codes
        self.code_dim = code_dim
        self.decay = decay
        self.eps = eps
        self.commitment_cost = commitment_cost
        self.diversity_weight = diversity_weight
        
        # Track whether codebook has been initialized with k-means
        self._kmeans_initialized = False
        
        # Initialize codebook embedding (normalize to unit norm to match cosine similarity)
        embedding = torch.randn(num_codes, code_dim)
        embedding = embedding / (embedding.norm(dim=1, keepdim=True) + self.eps)
        self.register_buffer("embedding", embedding)
        
        # EMA tracking buffers
        self.register_buffer("cluster_size", torch.zeros(num_codes))
        self.register_buffer("embed_avg", embedding.clone())
    
    @torch.no_grad()
    def init_codebook_kmeans(
        self,
        data: torch.Tensor,
        n_iters: int = 20,
        verbose: bool = False,
    ):
        """
        Initialize codebook using spherical k-means (cosine similarity).
        
        This warm-starts the codebook with cluster centroids computed from
        actual encoder outputs, leading to better initial code utilization.
        
        Args:
            data: Tensor of shape [N, D] containing encoder outputs to cluster.
            n_iters: Number of k-means iterations.
            verbose: Whether to print progress.
        
        Returns:
            Final cluster assignments [N].
        """
        N, D = data.shape
        device = data.device
        
        if N < self.num_codes:
            print(f"Warning: data samples ({N}) < num_codes ({self.num_codes}). "
                  f"Using random init for remaining codes.")
        
        # Normalize data for spherical k-means
        data_norm = data / (data.norm(dim=1, keepdim=True) + self.eps)  # [N, D]
        
        # Initialize centroids using k-means++ style initialization
        centroids = self._kmeans_plusplus_init(data_norm)
        
        # Run spherical k-means iterations
        for i in range(n_iters):
            # Compute cosine similarities: [N, num_codes]
            similarities = data_norm @ centroids.t()
            
            # Assign each point to nearest centroid
            assignments = similarities.argmax(dim=1)  # [N]
            
            # Update centroids
            new_centroids = torch.zeros_like(centroids)
            counts = torch.zeros(self.num_codes, device=device)
            
            for k in range(self.num_codes):
                mask = assignments == k
                if mask.sum() > 0:
                    # Mean of assigned points (unnormalized data for proper averaging)
                    new_centroids[k] = data[mask].mean(dim=0)
                    counts[k] = mask.sum()
                else:
                    # Keep old centroid if no points assigned
                    new_centroids[k] = centroids[k]
            
            # Normalize centroids to unit norm
            centroids = new_centroids / (new_centroids.norm(dim=1, keepdim=True) + self.eps)
            
            if verbose:
                # Compute usage stats
                used = (counts > 0).sum().item()
                print(f"K-means iter {i+1}/{n_iters}: {used}/{self.num_codes} codes used")
        
        # Set the codebook to the final centroids
        self.embedding.data.copy_(centroids)
        
        # Initialize EMA buffers based on final assignments
        final_similarities = data_norm @ centroids.t()
        final_assignments = final_similarities.argmax(dim=1)
        
        # Reset cluster sizes based on initial assignments
        for k in range(self.num_codes):
            mask = final_assignments == k
            self.cluster_size.data[k] = mask.sum().float()
            if mask.sum() > 0:
                self.embed_avg.data[k] = data[mask].mean(dim=0)
            else:
                self.embed_avg.data[k] = centroids[k]
        
        # Normalize embed_avg to be consistent
        self.embed_avg.data.copy_(
            self.embedding.data * self.cluster_size.data.unsqueeze(1)
        )
        
        self._kmeans_initialized = True
        
        if verbose:
            used_codes = (self.cluster_size > 0).sum().item()
            print(f"K-means init complete: {used_codes}/{self.num_codes} codes have assignments")
        
        return final_assignments
    
    @torch.no_grad()
    def _kmeans_plusplus_init(self, data_norm: torch.Tensor) -> torch.Tensor:
        """
        K-means++ initialization for better starting centroids.
        
        Args:
            data_norm: Normalized data [N, D].
        
        Returns:
            Initial centroids [num_codes, D].
        """
        N, D = data_norm.shape
        device = data_norm.device
        
        centroids = torch.zeros(self.num_codes, D, device=device)
        
        # Pick first centroid randomly
        idx = torch.randint(0, N, (1,), device=device).item()
        centroids[0] = data_norm[idx]
        
        for k in range(1, self.num_codes):
            if k >= N:
                # If we have more codes than data points, use random vectors
                centroids[k] = torch.randn(D, device=device)
                centroids[k] = centroids[k] / (centroids[k].norm() + self.eps)
                continue
            
            # Compute similarities to nearest existing centroid
            # Using cosine similarity (higher = closer)
            sims = data_norm @ centroids[:k].t()  # [N, k]
            max_sims = sims.max(dim=1).values  # [N]
            
            # Convert to distance-like measure (lower = farther from existing centroids)
            # We want to sample points that are far from existing centroids
            distances = 1 - max_sims  # [N]
            
            # Sample proportional to distance squared
            probs = distances ** 2
            probs = probs / probs.sum()
            
            # Sample next centroid
            idx = torch.multinomial(probs, 1).item()
            centroids[k] = data_norm[idx]
        
        return centroids
    
    def forward(self, z: torch.Tensor):
        """
        Args:
            z: Input tensor [B, D].
        
        Returns:
            Tuple of (z_q_st, vq_loss, perplexity, indices).
        """
        B, D = z.shape
        
        # Use cosine similarity for nearest code selection (semantic similarity)
        # Normalize inputs and embeddings then take argmax of dot product.
        # z: [B, D], embedding: [num_codes, D]
        z_norm = z / (z.norm(dim=1, keepdim=True) + self.eps)  # [B, D]
        emb_norm = self.embedding / (self.embedding.norm(dim=1, keepdim=True) + self.eps)  # [num_codes, D]
        similarities = z_norm @ emb_norm.t()  # [B, num_codes]

        # Find most similar code for each input
        indices = similarities.argmax(dim=1)  # [B]
        
        # Get quantized vectors
        z_q = F.embedding(indices, self.embedding)  # [B, D]
        
        # EMA update during training
        if self.training:
            # One-hot encoding for indices
            encodings = F.one_hot(indices, self.num_codes).float()  # [B, num_codes]
            
            # Update cluster sizes
            cluster_size = encodings.sum(dim=0)  # [num_codes]
            self.cluster_size.data.mul_(self.decay).add_(cluster_size, alpha=1 - self.decay)
            
            # Update embedding averages
            embed_sum = encodings.t() @ z  # [num_codes, D]
            self.embed_avg.data.mul_(self.decay).add_(embed_sum, alpha=1 - self.decay)
            
            # Normalize to update embeddings
            n = self.cluster_size.sum()
            cluster_size_normalized = (
                (self.cluster_size + self.eps) / (n + self.num_codes * self.eps) * n
            )
            self.embedding.data.copy_(self.embed_avg / cluster_size_normalized.unsqueeze(1))

            # After EMA update, normalize the codebook vectors to unit norm so that
            # cosine similarity (used for assignment) remains consistent.
            emb_norms = self.embedding.data.norm(dim=1, keepdim=True)
            self.embedding.data.div_(emb_norms + self.eps)
            
            # Also update embed_avg to stay consistent with the normalized embeddings
            # (prevents drift between embed_avg and embedding over time)
            self.embed_avg.data.copy_(self.embedding.data * cluster_size_normalized.unsqueeze(1))
        
        commitment_loss = F.mse_loss(z, z_q.detach())
        embedding_loss = F.mse_loss(z.detach(), z_q)
        
        counts = torch.bincount(indices, minlength=self.num_codes).float()  # [num_codes]
        probs = counts / counts.sum()  # [num_codes]
        entropy = -torch.sum(probs * torch.log(probs + 1e-10))
        max_entropy = math.log(self.num_codes)
        
        # Diversity loss: minimize (max_entropy - entropy) to maximize entropy
        diversity_loss = (max_entropy - entropy) / max_entropy
        
        vq_loss = commitment_loss * self.commitment_cost + embedding_loss + self.diversity_weight * diversity_loss
        z_q_st = z + (z_q - z).detach()
        
        # Compute perplexity from code usage histogram (hard assignments)
        encodings = F.one_hot(indices, self.num_codes).float()
        avg_probs = encodings.mean(dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))
        
        return z_q_st, vq_loss, perplexity, indices, diversity_loss


class KmerVQVAE(nn.Module):
    """
    VQ-VAE model for k-mer reconstruction.
    
    Encodes a token window, quantizes the pooled k-mer representation,
    and reconstructs the central k-mer tokens.
    """
    
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 256,
        n_layers: int = 4,
        n_heads: int = 4,
        max_len: int = 512,
        num_codes: int = 2048,
        code_dim: int = 256,
    ):
        """
        Args:
            vocab_size: Size of vocabulary.
            d_model: Model dimension.
            n_layers: Number of transformer layers.
            n_heads: Number of attention heads.
            max_len: Maximum sequence length.
            num_codes: Number of codes in VQ codebook.
            code_dim: Dimension of code vectors.
        """
        super().__init__()
        assert d_model == code_dim, f"d_model ({d_model}) must equal code_dim ({code_dim})"
        
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.max_len = max_len
        self.num_codes = num_codes
        self.code_dim = code_dim
        
        # Encoder
        self.encoder = TransformerEncoder(
            vocab_size=vocab_size,
            d_model=d_model,
            n_layers=n_layers,
            n_heads=n_heads,
            max_len=max_len,
        )
        
        # Vector Quantizer
        self.vq = VectorQuantizerEMA(num_codes=num_codes, code_dim=code_dim)
        
        # Decoder (simple linear projection to vocab)
        self.decoder = nn.Linear(code_dim, vocab_size)
    
    def forward(
        self,
        token_ids: torch.Tensor,
        kmer_start: torch.Tensor,
        kmer_len: int,
    ):
        """
        Args:
            token_ids: [B, T] tensor of token IDs.
            kmer_start: [B] tensor of k-mer start positions.
            kmer_len: Length of k-mer (scalar).
        
        Returns:
            Tuple of (logits, vq_loss, perplexity, indices, diversity_loss).
            - logits: [B, kmer_len, vocab_size]
            - vq_loss: scalar VQ loss (includes diversity regularization)
            - perplexity: scalar perplexity
            - indices: [B] codebook indices
            - diversity_loss: scalar diversity loss for logging
        """
        B, T = token_ids.shape
        device = token_ids.device
        
        # Encode the full window
        h = self.encoder(token_ids)  # [B, T, D]
        
        # Build positions for each k-mer
        # kmer_positions[i, j] = kmer_start[i] + j for j in 0..kmer_len-1
        kmer_offsets = torch.arange(kmer_len, device=device)  # [kmer_len]
        kmer_positions = kmer_start.unsqueeze(1) + kmer_offsets.unsqueeze(0)  # [B, kmer_len]
        
        # Gather k-mer hidden states
        # Expand positions for gathering: [B, kmer_len, D]
        kmer_positions_expanded = kmer_positions.unsqueeze(-1).expand(-1, -1, self.d_model)
        kmer_h = torch.gather(h, dim=1, index=kmer_positions_expanded)  # [B, kmer_len, D]
        
        # Pool across k-mer dimension (mean pooling)
        z = kmer_h.mean(dim=1)  # [B, D]
        
        # Quantize
        z_q_st, vq_loss, perplexity, indices, diversity_loss = self.vq(z)
        
        # Broadcast z_q_st to [B, kmer_len, D] and decode
        z_q_broadcast = z_q_st.unsqueeze(1).expand(-1, kmer_len, -1)  # [B, kmer_len, D]
        logits = self.decoder(z_q_broadcast)  # [B, kmer_len, vocab_size]
        
        return logits, vq_loss, perplexity, indices, diversity_loss
    
    @torch.no_grad()
    def warm_start_codebook(
        self,
        dataloader,
        kmer_len: int,
        max_samples: int = 50000,
        n_iters: int = 20,
        verbose: bool = True,
    ):
        """
        Warm-start the VQ codebook using k-means on encoder outputs.
        
        This should be called before training to initialize the codebook
        with meaningful cluster centroids from actual data.
        
        Args:
            dataloader: DataLoader yielding (token_ids, kmer_start) batches.
            kmer_len: Length of k-mer.
            max_samples: Maximum number of samples to collect for k-means.
            n_iters: Number of k-means iterations.
            verbose: Whether to print progress.
        
        Returns:
            Cluster assignments for the collected samples.
        """
        self.eval()
        device = next(self.parameters()).device
        
        # Collect encoder outputs
        z_list = []
        n_collected = 0
        
        if verbose:
            print(f"Collecting encoder outputs for k-means initialization...")
        
        for batch in dataloader:
            if n_collected >= max_samples:
                break
            
            # Handle different batch formats
            if isinstance(batch, (list, tuple)):
                token_ids = batch[0]
                kmer_start = batch[1] if len(batch) > 1 else None
            else:
                token_ids = batch
                kmer_start = None
            
            token_ids = token_ids.to(device)
            B, T = token_ids.shape
            
            # Default kmer_start to center if not provided
            if kmer_start is None:
                kmer_start = torch.full((B,), (T - kmer_len) // 2, device=device)
            else:
                kmer_start = kmer_start.to(device)
            
            # Get encoder outputs
            h = self.encoder(token_ids)  # [B, T, D]
            
            # Extract k-mer representations
            kmer_offsets = torch.arange(kmer_len, device=device)
            kmer_positions = kmer_start.unsqueeze(1) + kmer_offsets.unsqueeze(0)
            kmer_positions_expanded = kmer_positions.unsqueeze(-1).expand(-1, -1, self.d_model)
            kmer_h = torch.gather(h, dim=1, index=kmer_positions_expanded)
            z = kmer_h.mean(dim=1)  # [B, D]
            
            z_list.append(z.cpu())
            n_collected += B
        
        # Concatenate all collected embeddings
        all_z = torch.cat(z_list, dim=0)[:max_samples].to(device)
        
        if verbose:
            print(f"Collected {all_z.shape[0]} samples, running k-means...")
        
        # Run k-means initialization
        assignments = self.vq.init_codebook_kmeans(
            all_z,
            n_iters=n_iters,
            verbose=verbose,
        )
        
        return assignments


if __name__ == "__main__":
    # Test the model
    vocab_size = 50257  # GPT-2 vocab size
    batch_size = 4
    seq_len = 256
    kmer_len = 16
    
    model = KmerVQVAE(
        vocab_size=vocab_size,
        d_model=256,
        n_layers=4,
        n_heads=4,
        max_len=512,
        num_codes=2048,
        code_dim=256,
    )
    
    # Create dummy inputs
    token_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
    kmer_start = torch.randint(64, 192, (batch_size,))  # Centered k-mer starts
    
    # Forward pass (before k-means init)
    print("=== Before K-means Initialization ===")
    logits, vq_loss, perplexity, indices, diversity_loss = model(token_ids, kmer_start, kmer_len)
    
    print(f"logits shape: {logits.shape}")  # [B, kmer_len, vocab_size]
    print(f"vq_loss: {vq_loss.item():.4f}")
    print(f"perplexity: {perplexity.item():.2f}")
    print(f"diversity_loss: {diversity_loss.item():.4f}")
    print(f"indices shape: {indices.shape}")  # [B]
    
    # Test K-means warm start initialization
    print("\n=== K-means Warm Start Initialization ===")
    
    # Create a simple dummy dataloader for testing
    class DummyDataset(torch.utils.data.Dataset):
        def __init__(self, vocab_size, seq_len, n_samples=1000):
            self.vocab_size = vocab_size
            self.seq_len = seq_len
            self.n_samples = n_samples
        
        def __len__(self):
            return self.n_samples
        
        def __getitem__(self, idx):
            token_ids = torch.randint(0, self.vocab_size, (self.seq_len,))
            kmer_start = torch.randint(64, 192, (1,)).item()
            return token_ids, kmer_start
    
    dummy_dataset = DummyDataset(vocab_size, seq_len, n_samples=500)
    dummy_loader = torch.utils.data.DataLoader(
        dummy_dataset, batch_size=32, shuffle=True
    )
    
    # Warm start the codebook
    assignments = model.warm_start_codebook(
        dummy_loader,
        kmer_len=kmer_len,
        max_samples=500,
        n_iters=10,
        verbose=True,
    )
    
    # Forward pass (after k-means init)
    print("\n=== After K-means Initialization ===")
    model.train()
    logits, vq_loss, perplexity, indices, diversity_loss = model(token_ids, kmer_start, kmer_len)
    
    print(f"logits shape: {logits.shape}")
    print(f"vq_loss: {vq_loss.item():.4f}")
    print(f"perplexity: {perplexity.item():.2f}")
    print(f"diversity_loss: {diversity_loss.item():.4f}")
    print(f"indices: {indices.tolist()}")
    
    # Check codebook utilization
    used_codes = (model.vq.cluster_size > 0).sum().item()
    print(f"\nCodebook utilization: {used_codes}/{model.num_codes} codes used")
