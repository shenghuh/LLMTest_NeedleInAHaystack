import os
import torch
import torch.nn.functional as F

class Filter:
	def __init__(self, model):
		self.model = model
		self.chunk_embeddings = None
		self.chunks = None

	def _split_into_chunks(self, text: str, chunk_size: int = 512):
		# word-based splitter
		words = text.split()
		for i in range(0, self.chunk_embeddings.size(0)):
			start = i * chunk_size
			end = start + chunk_size
			chunk_words = words[start:end]
			if not chunk_words:
				break
			chunk_text = " ".join(chunk_words)
			self.chunks.append(chunk_text)
			chunk_emb = self.model.encode(chunk_text)  # [dim]
			self.chunk_embeddings[:,i] = chunk_emb
	
	def process_context(self, full_context: str, chunk_size: int = 512):
		self.chunks = []
		if len(full_context) % chunk_size == 0:
			self.chunk_embeddings = torch.empty((len(full_context)//chunk_size, self.model.get_sentence_embedding_dimension()))
		else:
			self.chunk_embeddings = torch.empty((len(full_context)//chunk_size + 1, self.model.get_sentence_embedding_dimension()))

		self._split_into_chunks(full_context, chunk_size=chunk_size)
		if self.chunk_embeddings:
			self.chunk_embeddings = torch.stack(self.chunk_embeddings, dim=0)  # [num_chunks, dim]
		else:
			self.chunk_embeddings = torch.empty((0, self.model.get_sentence_embedding_dimension()))

	def filter_context(self, question: str, chunk_size: int = 512, top_k: int = 3) -> str:
		if not self.chunks:
			return "".join(self.chunks)

		q_emb = self.model.encode(question)

		scores = torch.matmul(self.chunk_embeddings, q_emb)  # [num_chunks]
		k = min(top_k, len(self.chunks))
		topk = torch.topk(scores, k=k)
		idxs = topk.indices.tolist()

		selected_chunks = [self.chunks[i] for i in idxs]
		filtered_context = "\n\n".join(selected_chunks)
		return filtered_context    
