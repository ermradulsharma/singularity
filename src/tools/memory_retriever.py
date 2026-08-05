import torch
import hashlib
from src.memory import IndependentNeuralMemory

def _text_to_tensor(text: str, dim: int = 128) -> torch.Tensor:
    """
    Converts text to a dummy deterministic tensor for independent KV searching.
    In a full LLM setup, this would use the tokenizer/embedding layer.
    """
    # Deterministic hash array based on text chunks
    words = text.split()
    tensor_data = []
    for i in range(dim):
        if i < len(words):
            # Hash each word to a float between -1 and 1
            h = int(hashlib.md5(words[i].encode()).hexdigest()[:8], 16)
            tensor_data.append((h / 0xffffffff) * 2 - 1)
        else:
            tensor_data.append(0.0)
            
    return torch.tensor([tensor_data], dtype=torch.float32)

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
