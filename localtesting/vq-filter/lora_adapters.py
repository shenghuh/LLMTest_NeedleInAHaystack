"""
LoRA (Low-Rank Adaptation) module for efficient fine-tuning.

This module provides LoRA layers that can be injected into the VQ-VAE encoder
for domain adaptation without modifying the base model weights.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Dict, List, Tuple


class LoRALinear(nn.Module):
    """
    Linear layer with LoRA adaptation.
    
    Implements: output = Wx + (BA)x * scaling
    Where W is frozen, and B, A are low-rank trainable matrices.
    """
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int = 8,
        alpha: float = 16.0,
        dropout: float = 0.0,
        merge_weights: bool = False,
    ):
        """
        Args:
            in_features: Input dimension.
            out_features: Output dimension.
            rank: LoRA rank (r). Lower = fewer parameters.
            alpha: LoRA scaling factor. scaling = alpha / rank.
            dropout: Dropout probability for LoRA path.
            merge_weights: If True, merge LoRA into base weights (for inference).
        """
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        self.merge_weights = merge_weights
        
        # Base linear layer (frozen during LoRA training)
        self.linear = nn.Linear(in_features, out_features)
        
        # LoRA matrices
        self.lora_A = nn.Parameter(torch.zeros(rank, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))
        
        # Dropout
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        
        # Initialize LoRA weights
        self.reset_lora_parameters()
        
        # Track if weights are merged
        self.merged = False
    
    def reset_lora_parameters(self):
        """Initialize LoRA matrices."""
        # A uses Kaiming uniform, B starts at zero
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)
    
    def merge(self):
        """Merge LoRA weights into base linear layer."""
        if not self.merged:
            self.linear.weight.data += (self.lora_B @ self.lora_A) * self.scaling
            self.merged = True
    
    def unmerge(self):
        """Unmerge LoRA weights from base linear layer."""
        if self.merged:
            self.linear.weight.data -= (self.lora_B @ self.lora_A) * self.scaling
            self.merged = False
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.merged:
            return self.linear(x)
        else:
            # Base output + LoRA output
            base_out = self.linear(x)
            lora_out = (self.dropout(x) @ self.lora_A.T @ self.lora_B.T) * self.scaling
            return base_out + lora_out
    
    def freeze_base(self):
        """Freeze the base linear layer weights."""
        self.linear.weight.requires_grad = False
        if self.linear.bias is not None:
            self.linear.bias.requires_grad = False
    
    def unfreeze_base(self):
        """Unfreeze the base linear layer weights."""
        self.linear.weight.requires_grad = True
        if self.linear.bias is not None:
            self.linear.bias.requires_grad = True


class LoRAAttention(nn.Module):
    """
    Multi-head attention with LoRA on Q, K, V, and output projections.
    """
    
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        rank: int = 8,
        alpha: float = 16.0,
        dropout: float = 0.1,
        lora_dropout: float = 0.0,
        apply_lora_to: List[str] = ["q", "v"],  # Which projections get LoRA
    ):
        """
        Args:
            d_model: Model dimension.
            n_heads: Number of attention heads.
            rank: LoRA rank.
            alpha: LoRA scaling factor.
            dropout: Attention dropout.
            lora_dropout: LoRA-specific dropout.
            apply_lora_to: List of projections to apply LoRA to ["q", "k", "v", "o"].
        """
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.apply_lora_to = apply_lora_to
        
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        
        # Q, K, V projections (with optional LoRA)
        self.q_proj = self._make_proj("q", d_model, d_model, rank, alpha, lora_dropout)
        self.k_proj = self._make_proj("k", d_model, d_model, rank, alpha, lora_dropout)
        self.v_proj = self._make_proj("v", d_model, d_model, rank, alpha, lora_dropout)
        self.o_proj = self._make_proj("o", d_model, d_model, rank, alpha, lora_dropout)
        
        self.dropout = nn.Dropout(dropout)
        self.scale = self.head_dim ** -0.5
    
    def _make_proj(self, name, in_f, out_f, rank, alpha, dropout):
        """Create projection layer with or without LoRA."""
        if name in self.apply_lora_to:
            return LoRALinear(in_f, out_f, rank=rank, alpha=alpha, dropout=dropout)
        else:
            return nn.Linear(in_f, out_f)
    
    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B, T, D = x.shape
        
        # Project to Q, K, V
        q = self.q_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        
        # Attention scores
        attn = (q @ k.transpose(-2, -1)) * self.scale
        
        if mask is not None:
            attn = attn.masked_fill(mask == 0, float('-inf'))
        
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        
        # Apply attention to values
        out = (attn @ v).transpose(1, 2).contiguous().view(B, T, D)
        
        return self.o_proj(out)
    
    def freeze_base(self):
        """Freeze all base (non-LoRA) weights."""
        for proj in [self.q_proj, self.k_proj, self.v_proj, self.o_proj]:
            if isinstance(proj, LoRALinear):
                proj.freeze_base()
            else:
                for param in proj.parameters():
                    param.requires_grad = False


class ProjectionAdapter(nn.Module):
    """
    Simple projection adapter for task-specific fine-tuning.
    Adds a residual MLP that transforms representations.
    """
    
    def __init__(
        self,
        d_model: int,
        hidden_dim: Optional[int] = None,
        dropout: float = 0.1,
        num_layers: int = 2,
    ):
        """
        Args:
            d_model: Input/output dimension.
            hidden_dim: Hidden layer dimension (default: 4 * d_model).
            dropout: Dropout probability.
            num_layers: Number of layers in the adapter.
        """
        super().__init__()
        hidden_dim = hidden_dim or 4 * d_model
        
        layers = []
        for i in range(num_layers):
            if i == 0:
                layers.append(nn.Linear(d_model, hidden_dim))
            elif i == num_layers - 1:
                layers.append(nn.Linear(hidden_dim, d_model))
            else:
                layers.append(nn.Linear(hidden_dim, hidden_dim))
            
            if i < num_layers - 1:
                layers.append(nn.GELU())
                layers.append(nn.Dropout(dropout))
        
        self.mlp = nn.Sequential(*layers)
        
        # Initialize last layer to near-zero for residual
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.mlp(x)


class DomainAdapter(nn.Module):
    """
    Domain-specific adapter that can be swapped for different domains.
    Uses AdaLN-style conditioning or simple projection.
    """
    
    def __init__(
        self,
        d_model: int,
        num_domains: int = 3,  # e.g., web, academic, conversational
        adapter_type: str = "projection",  # "projection" or "adaln"
    ):
        """
        Args:
            d_model: Model dimension.
            num_domains: Number of domains to adapt to.
            adapter_type: Type of adaptation ("projection" or "adaln").
        """
        super().__init__()
        self.d_model = d_model
        self.num_domains = num_domains
        self.adapter_type = adapter_type
        
        if adapter_type == "projection":
            # Per-domain projection adapters
            self.adapters = nn.ModuleList([
                ProjectionAdapter(d_model, hidden_dim=d_model, num_layers=2)
                for _ in range(num_domains)
            ])
        elif adapter_type == "adaln":
            # AdaLN-Zero style: learn scale and shift per domain
            self.domain_embed = nn.Embedding(num_domains, d_model * 2)
            nn.init.zeros_(self.domain_embed.weight)
        else:
            raise ValueError(f"Unknown adapter_type: {adapter_type}")
    
    def forward(self, x: torch.Tensor, domain_id: int = 0) -> torch.Tensor:
        if self.adapter_type == "projection":
            return self.adapters[domain_id](x)
        elif self.adapter_type == "adaln":
            # AdaLN: x = x * (1 + scale) + shift
            params = self.domain_embed(torch.tensor(domain_id, device=x.device))
            scale, shift = params.chunk(2, dim=-1)
            return x * (1 + scale) + shift


class QueryPassageAdapters(nn.Module):
    """
    Dual adapters for query-passage alignment.
    Separate adapters for query and passage representations.
    """
    
    def __init__(
        self,
        d_model: int,
        hidden_dim: Optional[int] = None,
        shared_base: bool = False,
        dropout: float = 0.1,
    ):
        """
        Args:
            d_model: Model dimension.
            hidden_dim: Hidden dimension for adapters.
            shared_base: If True, share first layer between query/passage.
            dropout: Dropout probability.
        """
        super().__init__()
        hidden_dim = hidden_dim or d_model
        
        if shared_base:
            self.shared_proj = nn.Linear(d_model, hidden_dim)
            self.query_head = nn.Sequential(
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, d_model),
            )
            self.passage_head = nn.Sequential(
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, d_model),
            )
        else:
            self.shared_proj = None
            self.query_head = ProjectionAdapter(d_model, hidden_dim, dropout)
            self.passage_head = ProjectionAdapter(d_model, hidden_dim, dropout)
        
        self.shared_base = shared_base
    
    def forward_query(self, x: torch.Tensor) -> torch.Tensor:
        """Transform query representations."""
        if self.shared_base:
            return x + self.query_head(self.shared_proj(x))
        else:
            return self.query_head(x)
    
    def forward_passage(self, x: torch.Tensor) -> torch.Tensor:
        """Transform passage representations."""
        if self.shared_base:
            return x + self.passage_head(self.shared_proj(x))
        else:
            return self.passage_head(x)


def add_lora_to_model(
    model: nn.Module,
    rank: int = 8,
    alpha: float = 16.0,
    target_modules: List[str] = ["q_proj", "v_proj"],
    dropout: float = 0.0,
) -> Tuple[nn.Module, Dict[str, nn.Parameter]]:
    """
    Add LoRA adapters to specified modules in a model.
    
    Args:
        model: The model to modify.
        rank: LoRA rank.
        alpha: LoRA scaling factor.
        target_modules: Names of modules to add LoRA to.
        dropout: LoRA dropout.
    
    Returns:
        Modified model and dict of LoRA parameters.
    """
    lora_params = {}
    
    for name, module in model.named_modules():
        if any(target in name for target in target_modules):
            if isinstance(module, nn.Linear):
                # Replace with LoRALinear
                parent_name = ".".join(name.split(".")[:-1])
                child_name = name.split(".")[-1]
                
                parent = model
                for part in parent_name.split("."):
                    if part:
                        parent = getattr(parent, part)
                
                lora_module = LoRALinear(
                    module.in_features,
                    module.out_features,
                    rank=rank,
                    alpha=alpha,
                    dropout=dropout,
                )
                # Copy base weights
                lora_module.linear.weight.data = module.weight.data.clone()
                if module.bias is not None:
                    lora_module.linear.bias.data = module.bias.data.clone()
                
                lora_module.freeze_base()
                setattr(parent, child_name, lora_module)
                
                lora_params[f"{name}.lora_A"] = lora_module.lora_A
                lora_params[f"{name}.lora_B"] = lora_module.lora_B
    
    return model, lora_params


def get_lora_parameters(model: nn.Module) -> List[nn.Parameter]:
    """Get only LoRA parameters from a model."""
    lora_params = []
    for name, param in model.named_parameters():
        if "lora_" in name:
            lora_params.append(param)
    return lora_params


def get_non_lora_parameters(model: nn.Module) -> List[nn.Parameter]:
    """Get non-LoRA parameters from a model."""
    non_lora_params = []
    for name, param in model.named_parameters():
        if "lora_" not in name:
            non_lora_params.append(param)
    return non_lora_params


def freeze_base_model(model: nn.Module, freeze_codebook: bool = True):
    """Freeze all base model parameters, keeping LoRA trainable."""
    for name, param in model.named_parameters():
        if "lora_" in name:
            param.requires_grad = True
        elif "adapter" in name:
            param.requires_grad = True
        elif not freeze_codebook and "codebook" in name:
            param.requires_grad = True
        else:
            param.requires_grad = False


def count_parameters(model: nn.Module) -> Dict[str, int]:
    """Count trainable and total parameters."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    lora = sum(p.numel() for n, p in model.named_parameters() if "lora_" in n)
    adapter = sum(p.numel() for n, p in model.named_parameters() if "adapter" in n)
    
    return {
        "total": total,
        "trainable": trainable,
        "frozen": total - trainable,
        "lora": lora,
        "adapter": adapter,
        "trainable_pct": 100 * trainable / total if total > 0 else 0,
    }


if __name__ == "__main__":
    # Test LoRA modules
    print("Testing LoRA modules...")
    
    # Test LoRALinear
    lora_linear = LoRALinear(384, 384, rank=8, alpha=16)
    x = torch.randn(2, 64, 384)
    out = lora_linear(x)
    print(f"LoRALinear: input {x.shape} -> output {out.shape}")
    
    # Test ProjectionAdapter
    adapter = ProjectionAdapter(384, hidden_dim=384)
    out = adapter(x)
    print(f"ProjectionAdapter: input {x.shape} -> output {out.shape}")
    
    # Test QueryPassageAdapters
    qp_adapters = QueryPassageAdapters(384)
    q_out = qp_adapters.forward_query(x)
    p_out = qp_adapters.forward_passage(x)
    print(f"QueryPassageAdapters: query {q_out.shape}, passage {p_out.shape}")
    
    # Count parameters
    print(f"\nLoRALinear parameters: {count_parameters(lora_linear)}")
    print(f"ProjectionAdapter parameters: {count_parameters(adapter)}")
    
    print("\nLoRA module tests passed!")
