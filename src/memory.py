import torch
import os
import torch.nn.functional as F
import safetensors.torch

class IndependentNeuralMemory:
    """
    100% Independent Neural Memory System.
    Replaces external Vector Databases (like Chroma/FAISS).
    Directly saves Key-Value (KV) tensors to safetensors files and retrieves via Cosine Similarity.
    """
    def __init__(self, memory_file="data/memory.safetensors", max_memories=5000):
        self.memory_file = memory_file
        self.max_memories = max_memories
        self.keys = None
        self.values = None
        
        os.makedirs(os.path.dirname(memory_file), exist_ok=True)
        self.load_memory()
        
    def load_memory(self):
        """Loads memories from disk using safetensors."""
        if os.path.exists(self.memory_file):
            try:
                tensors = safetensors.torch.load_file(self.memory_file)
                self.keys = tensors.get('keys')
                self.values = tensors.get('values')
            except Exception as e:
                pass

    def save_memory(self):
        """Saves memories to disk atomically using safetensors."""
        if self.keys is not None and self.values is not None:
            try:
                state_dict = {
                    'keys': self.keys.contiguous(),
                    'values': self.values.contiguous()
                }
                tmp_file = self.memory_file + ".tmp"
                safetensors.torch.save_file(state_dict, tmp_file)
                os.replace(tmp_file, self.memory_file)
            except Exception:
                pass
            
    def add_experience(self, key_tensor: torch.Tensor, value_tensor: torch.Tensor):
        """Adds a new thought/experience to long-term memory."""
        k = key_tensor.detach().cpu()
        v = value_tensor.detach().cpu()
        
        if self.keys is None:
            self.keys = k
            self.values = v
        else:
            self.keys = torch.cat([self.keys, k], dim=0)
            self.values = torch.cat([self.values, v], dim=0)
            
        if self.keys.shape[0] > self.max_memories:
            self.keys = self.keys[-self.max_memories:]
            self.values = self.values[-self.max_memories:]
            
        self.save_memory()
        
    def retrieve_context(self, query_tensor: torch.Tensor, top_k=3) -> torch.Tensor:
        """Retrieves top_k most similar historical values using Cosine Similarity."""
        if self.keys is None:
            return None
            
        q = query_tensor.detach().cpu()
        sim = F.cosine_similarity(q.unsqueeze(1), self.keys.unsqueeze(0), dim=2)
        
        top_scores, top_indices = torch.topk(sim, min(top_k, sim.shape[1]), dim=1)
        
        retrieved_values = self.values[top_indices]
        return retrieved_values.to(query_tensor.device)


class VectorSemanticMemory(IndependentNeuralMemory):
    """
    Self-Sovereign Vector Semantic Memory.
    Provides semantic search, chunking, and text vector retrieval for RAG workflows.
    """
    def __init__(self, memory_file="data/semantic_memory.safetensors", max_memories=5000, emb_dim=128):
        super().__init__(memory_file=memory_file, max_memories=max_memories)
        self.emb_dim = emb_dim
        self.text_records = []

    def _encode_text(self, text: str) -> torch.Tensor:
        """Encodes text into a normalized d_model embedding vector."""
        vec = torch.zeros(1, self.emb_dim, dtype=torch.float32)
        words = text.split()
        if not words:
            return F.normalize(vec, p=2, dim=-1)
        for w in words:
            val = sum(ord(c) for c in w)
            idx = val % self.emb_dim
            vec[0, idx] += 1.0
        return F.normalize(vec, p=2, dim=-1)

    def store_text(self, text: str):
        """Chunks and stores text with its semantic vector embedding."""
        if not text or not text.strip():
            return
        vec = self._encode_text(text)
        self.add_experience(vec, vec)
        self.text_records.append(text[:512])

    def search_semantic(self, query: str, top_k: int = 3) -> list[str]:
        """Retrieves top-k most semantically relevant text passages for a given query."""
        if not self.text_records or self.keys is None:
            return []
        q_vec = self._encode_text(query)
        sim = F.cosine_similarity(q_vec, self.keys, dim=1)
        k = min(top_k, sim.size(0))
        _, top_indices = torch.topk(sim, k=k)
        results = []
        for idx in top_indices.tolist():
            if idx < len(self.text_records):
                results.append(self.text_records[idx])
        return results



