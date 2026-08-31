import torch
import hashlib
from src.memory import IndependentNeuralMemory

def _text_to_tensor(text: str, dim: int = 128) -> torch.Tensor:
    """Converts text to a real deterministic embedding tensor using a Term Frequency (TF) hashing trick vectorizer."""
    import math
    import collections
    
    words = text.lower().replace('.', ' ').replace(',', ' ').split()
    word_counts = collections.Counter(words)
    
    tensor_data = [0.0] * dim
    
    for word, count in word_counts.items():
        h_hex = hashlib.md5(word.encode()).hexdigest()
        idx = int(h_hex, 16) % dim
        
        tf = 1 + math.log(count)
        tensor_data[idx] += tf
        
    tensor_vec = torch.tensor(tensor_data, dtype=torch.float32)
    norm = torch.norm(tensor_vec, p=2)
    if norm > 0:
        tensor_vec = tensor_vec / norm
        
    return tensor_vec.unsqueeze(0)

from src.memory import VectorSemanticMemory

_GLOBAL_SEMANTIC_MEMORY = None

def _get_semantic_memory() -> VectorSemanticMemory:
    global _GLOBAL_SEMANTIC_MEMORY
    if _GLOBAL_SEMANTIC_MEMORY is None:
        _GLOBAL_SEMANTIC_MEMORY = VectorSemanticMemory()
    return _GLOBAL_SEMANTIC_MEMORY

def store_in_memory(key_text: str, value_text: str):
    """Stores a concept (key) and its definition (value) into long-term memory."""
    mem = _get_semantic_memory()
    combined_text = f"[{key_text}]: {value_text}"
    mem.store_text(combined_text)
    return "Stored in Long-Term Memory successfully."

def search_memory(query_text: str, top_k: int = 3) -> str:
    """Searches the agent's long term memory and returns actual text content passages."""
    mem = _get_semantic_memory()
    passages = mem.search_semantic(query_text, top_k=top_k)
    
    if not passages:
        return "No relevant memories found."
        
    return " | ".join(passages)

