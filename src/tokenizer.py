"""
Unified Latest-Generation Tokenizer Module for Singularity AGI Engine.
Provides consistent encoding/decoding using 4th-Gen Tokenizers (o200k_base / cl100k_base).
"""

import tiktoken
from typing import List, Union

_TOKENIZER_CACHE = None

class UnifiedTokenizer:
    """Wrapper for 4th-Gen Tiktoken Encoders with fallback resilience."""
    def __init__(self, encoding_name: str = "o200k_base"):
        self.encoding_name = encoding_name
        self.enc = self._init_encoding(encoding_name)
        self.n_vocab = getattr(self.enc, "n_vocab", 200019)

    def _init_encoding(self, preferred: str):
        candidates = [preferred, "o200k_base", "cl100k_base", "gpt2"]
        for cand in candidates:
            try:
                return tiktoken.get_encoding(cand)
            except Exception:
                continue
        raise RuntimeError("Failed to load any tiktoken encoding scheme.")

    def encode(self, text: str, allowed_special: Union[str, set] = "all", disallowed_special: Union[str, set] = ()) -> List[int]:
        """Encodes text to token IDs safely."""
        try:
            return self.enc.encode(text, allowed_special=allowed_special, disallowed_special=disallowed_special)
        except Exception:
            return self.enc.encode(text)

    def decode(self, tokens: List[int], errors: str = "replace") -> str:
        """Decodes token IDs back to string."""
        valid_tokens = [int(t) for t in tokens if 0 <= int(t) < self.n_vocab]
        try:
            return self.enc.decode(valid_tokens, errors=errors)
        except Exception:
            return self.enc.decode(valid_tokens)

def get_unified_tokenizer(encoding_name: str = "o200k_base") -> UnifiedTokenizer:
    """Returns a singleton instance of the UnifiedTokenizer."""
    global _TOKENIZER_CACHE
    if _TOKENIZER_CACHE is None:
        _TOKENIZER_CACHE = UnifiedTokenizer(encoding_name=encoding_name)
    return _TOKENIZER_CACHE
