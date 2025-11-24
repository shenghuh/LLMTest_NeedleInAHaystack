import os
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

from .clip_models import CLLP

class CLLPFilter:
    def __init__(self, ckpt_path: str = "models/cllp_final.pth", d_model: int = 256,
                 latent_dim: int = 64, nhead: int = 4, nlayers: int = 2,
                 device: str = None):
        base_dir = os.path.dirname(__file__)
        if not os.path.isabs(ckpt_path):
            ckpt_path = os.path.join(base_dir, ckpt_path)

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained("gpt2")
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        if self.tokenizer.cls_token is None:
            self.tokenizer.add_special_tokens({"cls_token": "[CLS]"})

        # Build model and load weights
        self.model = CLLP.from_pretrained(
            ckpt_path, self.tokenizer, d_model, latent_dim, nhead, nlayers
        ).to(self.device)
        self.model.eval()

    def _encode_text(self, text: str, max_length: int = 512) -> torch.Tensor:
        enc = self.tokenizer(
            text,
            max_length=max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        input_ids = enc["input_ids"].to(self.device)
        attn_mask = enc["attention_mask"].to(self.device)
        pad_mask = (attn_mask == 0)

        with torch.no_grad():
            emb = self.model(input_ids, pad_mask, cls_only=True)  # [1, dim]
            emb = F.normalize(emb, dim=-1)
        return emb.squeeze(0)  # [dim]

    def _split_into_chunks(self, text: str, max_tokens: int = 512, overlap: int = 128):
        # word-based splitter
        words = text.split()
        step = max_tokens - overlap
        chunks = []
        for start in range(0, len(words), step):
            end = start + max_tokens
            chunk_words = words[start:end]
            if not chunk_words:
                break
            chunks.append(" ".join(chunk_words))
        return chunks

    def filter_context(self, full_context: str, question: str, top_k: int = 3) -> str:
        chunks = self._split_into_chunks(full_context, max_tokens=512, overlap=128)

        if not chunks:
            return full_context  # fallback

        q_emb = self._encode_text(question, max_length=128)
        chunk_embs = torch.stack(
            [self._encode_text(ch, max_length=512) for ch in chunks], dim=0
        )  # [num_chunks, dim]

        scores = torch.matmul(chunk_embs, q_emb)  # [num_chunks]
        k = min(top_k, len(chunks))
        topk = torch.topk(scores, k=k)
        idxs = topk.indices.tolist()

        selected_chunks = [chunks[i] for i in idxs]
        filtered_context = "\n\n".join(selected_chunks)
        return filtered_context
