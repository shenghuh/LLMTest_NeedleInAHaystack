import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from datasets import load_dataset
import random

class ClipMiniPileDataset(Dataset):
	"""
	Dataset wrapper for MiniPile text data.
	Tokenizes text and creates fixed-length sequences.
	"""
	def __init__(self, tokenizer, split='train', max_length=512, query_max_length=128, max_samples=None):
		super(ClipMiniPileDataset, self).__init__()

		print(f"Loading MiniPile dataset ({split} split)...")
		# Load MiniPile dataset from Hugging Face
		self.dataset = load_dataset('JeanKaddour/minipile', split=split, streaming=False)

		if max_samples is not None:
			self.dataset = self.dataset.select(range(min(max_samples, len(self.dataset))))

		self.tokenizer = tokenizer
		if self.tokenizer.pad_token is None:
			self.tokenizer.pad_token = self.tokenizer.eos_token
		if self.tokenizer.cls_token is None:
			self.tokenizer.add_special_tokens({'cls_token': '[CLS]'})
			self.tokenizer.cls_token = '[CLS]'

		self.max_length = max_length
		self.query_max_length = query_max_length
		print(f"Dataset loaded: {len(self.dataset)} samples")
    
	def __len__(self):
		return len(self.dataset)
	
	def __getitem__(self, idx):
		text = self.dataset[idx]['text']
		encoded = self.tokenizer(
			text,
			max_length=self.max_length,
			padding='max_length',
			add_special_tokens=True,
			truncation=True,
			return_tensors='pt'
        )
		input_ids = encoded['input_ids'].squeeze(0)  # [max_length]
		attention_mask = encoded['attention_mask'].squeeze(0)  # [max_length]

		try:
			full_token_ids = self.tokenizer.encode(text, add_special_tokens=False, truncation=True, max_length=self.max_length)
		except Exception:
            # Fallback: if tokenizer.encode fails for any reason, treat as empty
			full_token_ids = []

		if len(full_token_ids) == 0:
            # Empty or un-tokenizable text: return a fully-padded label
			label_max_length = self.query_max_length
			label_input_ids = torch.full((label_max_length,), self.tokenizer.pad_token_id, dtype=torch.long)
			label_attention_mask = torch.zeros(label_max_length, dtype=torch.long)
		else:
			token_len = len(full_token_ids)
            # choose label length between 1 and min(self.query_max_length, token_len)
			max_label_len = min(self.query_max_length, token_len)
			label_len = random.randint(1, max_label_len)
			if token_len == label_len:
				start = 0
			else:
				start = random.randint(0, token_len - label_len)
			sub_tokens = full_token_ids[start:start + label_len]
			# Decode the token slice back to text and re-tokenize/pad to fixed 128
			sub_text = self.tokenizer.decode(sub_tokens, skip_special_tokens=True, clean_up_tokenization_spaces=True)
			encoded_label = self.tokenizer(
				sub_text,
				max_length=self.query_max_length,
				padding='max_length',
				add_special_tokens=True,
				truncation=True,
				return_tensors='pt'
			)
			label_input_ids = encoded_label['input_ids'].squeeze(0)
			label_attention_mask = encoded_label['attention_mask'].squeeze(0)
		return input_ids, attention_mask, label_input_ids, label_attention_mask

class CLLP(nn.Module):
	"""
	Contrastive Language-Language Pretraining Model
	"""
	def __init__(self,tokenizer,d_model,latent_dim,nhead,nlayers):
		super(CLLP, self).__init__()
		# Model initialization code here
		self.tokenizer = tokenizer
		vocab_size = len(self.tokenizer)
		self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=d_model, padding_idx=self.tokenizer.pad_token_id)
		self.encoder_layer = nn.TransformerEncoderLayer(d_model,nhead,batch_first=True,norm_first=True)
		self.encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=nlayers)
		self.fc = nn.Linear(d_model, latent_dim)

	def forward(self, x, attention_mask=None, cls_only=False):
		# Forward pass code here
		embedded = self.embedding(x)
		encoded_output = self.encoder(embedded, src_key_padding_mask=attention_mask)
		latent = self.fc(encoded_output)
		if cls_only:
			return latent[:,0,:]
		return latent
	
	@classmethod
	def from_pretrained(cls, path, tokenizer, d_model, latent_dim, nhead, nlayers):
		"""
		Load a pretrained model from a .pth file
		
		Args:
			path: Path to the .pth file
			tokenizer: Tokenizer instance
			d_model: Model dimension
			latent_dim: Latent dimension
			nhead: Number of attention heads
			nlayers: Number of transformer layers
		
		Returns:
			CLLP model with loaded weights
		"""
		model = cls(tokenizer, d_model, latent_dim, nhead, nlayers)
		model.load_state_dict(torch.load(path, map_location='cpu'))
		return model