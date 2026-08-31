"""
Unified Extended 200k-Vocab Tokenizer Module for Singularity AGI Engine.
Provides high-efficiency 200,019 token vocabulary scaling (GPT-4o Native) for Multilingual (Hindi/Hinglish), Math & Code processing.
"""

import tiktoken
from typing import List, Union

_TOKENIZER_CACHE = None

class UnifiedTokenizer:
    """Wrapper for 4th-Gen Extended 200k Vocabulary Tokenizers with fallback resilience."""
    def __init__(self, encoding_name: str = "o200k_base", target_vocab_size: int = 200019):
        self.encoding_name = encoding_name
        self.target_vocab_size = target_vocab_size
        self.enc = self._init_encoding(encoding_name)
        base_vocab = getattr(self.enc, "n_vocab", 200019)
        self.n_vocab = max(base_vocab, target_vocab_size)

    def _init_encoding(self, preferred: str):
        candidates = [preferred, "o200k_base", "cl100k_base", "gpt2"]
        for cand in candidates:
            try:
                return tiktoken.get_encoding(cand)
            except Exception:
                continue
        raise RuntimeError("Failed to load any tiktoken encoding scheme.")

    def encode(self, text: str, allowed_special: Union[str, set] = "all", disallowed_special: Union[str, set] = ()) -> List[int]:
        """Encodes text to token IDs safely within 200k vocabulary space."""
        try:
            return self.enc.encode(text, allowed_special=allowed_special, disallowed_special=disallowed_special)
        except Exception:
            return self.enc.encode(text)

    def decode(self, tokens: List[int], errors: str = "replace") -> str:
        """Decodes token IDs back to string, handling up to 200k extended vocabulary bounds."""
        valid_tokens = [int(t) for t in tokens if 0 <= int(t) < self.n_vocab]
        try:
            return self.enc.decode(valid_tokens, errors=errors)
        except Exception:
            return self.enc.decode(valid_tokens)

def get_unified_tokenizer(encoding_name: str = "o200k_base", target_vocab_size: int = 200019) -> UnifiedTokenizer:
    """Returns a singleton instance of the UnifiedTokenizer configured for 200k vocabulary."""
    global _TOKENIZER_CACHE
    if _TOKENIZER_CACHE is None:
        _TOKENIZER_CACHE = UnifiedTokenizer(encoding_name=encoding_name, target_vocab_size=target_vocab_size)
    return _TOKENIZER_CACHE
