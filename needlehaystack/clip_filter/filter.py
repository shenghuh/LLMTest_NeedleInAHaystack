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
        self.chunks = []
        embeddings_list = []
        
        for i in range(0, len(words), chunk_size):
            chunk_words = words[i:i + chunk_size]
            if not chunk_words:
                continue
            chunk_text = " ".join(chunk_words)
            self.chunks.append(chunk_text)
            chunk_emb = self.model.encode(chunk_text)  # [dim]
            embeddings_list.append(torch.as_tensor(chunk_emb))
        
        if embeddings_list:
            self.chunk_embeddings = torch.stack(embeddings_list)
        else:
            self.chunk_embeddings = None
    
    def process_context(self, full_context: str, chunk_size: int = 512):
        self._split_into_chunks(full_context, chunk_size=chunk_size)
    
    def filter_context(self, full_context: str, question: str, chunk_size: int = 512, top_k: int = 3) -> str:
        """
        Filter context to return only the most relevant chunks for the question.
        
        Args:
            full_context: The full context text to filter
            question: The question to use for relevance scoring
            chunk_size: Number of words per chunk
            top_k: Number of top chunks to return
            
        Returns:
            Filtered context containing the top-k most relevant chunks
        """
        # Process context into chunks
        self.process_context(full_context, chunk_size=chunk_size)
        
        if not self.chunks or self.chunk_embeddings is None:
            return full_context  # Return original if no chunks
        
        q_emb = torch.as_tensor(self.model.encode(question), device=self.chunk_embeddings.device)
        
        scores = torch.matmul(self.chunk_embeddings, q_emb)  # [num_chunks]
        k = min(top_k, len(self.chunks))
        topk = torch.topk(scores, k=k)
        idxs = sorted(topk.indices.tolist())  # Sort to maintain original order
        
        selected_chunks = [self.chunks[i] for i in idxs]
        filtered_context = "\n\n".join(selected_chunks)
        return filtered_context    
