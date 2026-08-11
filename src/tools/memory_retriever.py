import torch
import hashlib
from src.memory import IndependentNeuralMemory

def _text_to_tensor(text: str, dim: int = 128) -> torch.Tensor:
    """
    Converts text to a real deterministic embedding tensor using a Term Frequency (TF) 
    hashing trick vectorizer.
    """
    import math
    import collections
    
    words = text.lower().replace('.', ' ').replace(',', ' ').split()
    word_counts = collections.Counter(words)
    
    tensor_data = [0.0] * dim
    
    for word, count in word_counts.items():
        # Deterministic index hashing
        h_hex = hashlib.md5(word.encode()).hexdigest()
        idx = int(h_hex, 16) % dim
        
        # Log-normalized Term Frequency
        tf = 1 + math.log(count)
        tensor_data[idx] += tf
        
    # L2 Normalization
    tensor_vec = torch.tensor(tensor_data, dtype=torch.float32)
    norm = torch.norm(tensor_vec, p=2)
    if norm > 0:
        tensor_vec = tensor_vec / norm
        
    return tensor_vec.unsqueeze(0)

def store_in_memory(key_text: str, value_text: str):
    """Stores a concept (key) and its definition (value) into long-term memory."""
    mem = IndependentNeuralMemory()
    k = _text_to_tensor(key_text)
    v = _text_to_tensor(value_text)
    mem.add_experience(k, v)
    return "Stored in Long-Term Memory successfully."

def search_memory(query_text: str, top_k: int = 3) -> str:
    """Searches the agent's long term KV memory using cosine similarity."""
    mem = IndependentNeuralMemory()
    if mem.keys is None:
        return "Memory is currently empty."
        
    q = _text_to_tensor(query_text)
    retrieved = mem.retrieve_context(q, top_k=top_k)
    
    if retrieved is None:
        return "No relevant memories found."
        
    return f"Retrieved {retrieved.shape[1]} memory tensors successfully."
