import torch
import os
import torch.nn.functional as F

class IndependentNeuralMemory:
    """
    100% Independent Neural Memory System.
    Replaces external Vector Databases (like Chroma/FAISS).
    Directly saves Key-Value (KV) tensors to a binary file and retrieves via Cosine Similarity.
    """
    def __init__(self, memory_file="data/memory.agi", max_memories=5000):
        self.memory_file = memory_file
        self.max_memories = max_memories
        self.keys = None
        self.values = None
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(memory_file), exist_ok=True)
        self.load_memory()
        
    def load_memory(self):
        """Loads memories from disk."""
        if os.path.exists(self.memory_file):
            try:
                data = torch.load(self.memory_file, weights_only=True) # 🛡️ SECURITY PATCH
                self.keys = data['keys']
                self.values = data['values']
            except Exception as e:
                pass
    def save_memory(self):
        """Saves memories to disk."""
        if self.keys is not None:
            torch.save({'keys': self.keys, 'values': self.values}, self.memory_file)
            
    def add_experience(self, key_tensor: torch.Tensor, value_tensor: torch.Tensor):
        """Adds a new thought/experience to long-term memory."""
        # Detach and move to CPU to prevent memory leaks and graph issues
        k = key_tensor.detach().cpu()
        v = value_tensor.detach().cpu()
        
        if self.keys is None:
            self.keys = k
            self.values = v
        else:
            self.keys = torch.cat([self.keys, k], dim=0)
            self.values = torch.cat([self.values, v], dim=0)
            
        # Limit memory size
        if self.keys.shape[0] > self.max_memories:
            self.keys = self.keys[-self.max_memories:]
            self.values = self.values[-self.max_memories:]
            
        self.save_memory()
        
    def retrieve_context(self, query_tensor: torch.Tensor, top_k=3) -> torch.Tensor:
        """Retrieves top_k most similar historical values using Cosine Similarity."""
        if self.keys is None:
            return None
            
        q = query_tensor.detach().cpu()
        # Cosine similarity: (q @ k.T) / (|q||k|)
        # q shape: (batch, dim), keys shape: (num_memories, dim)
        sim = F.cosine_similarity(q.unsqueeze(1), self.keys.unsqueeze(0), dim=2)
        
        top_scores, top_indices = torch.topk(sim, min(top_k, sim.shape[1]), dim=1)
        
        retrieved_values = self.values[top_indices] # (batch, top_k, dim)
        return retrieved_values.to(query_tensor.device)
