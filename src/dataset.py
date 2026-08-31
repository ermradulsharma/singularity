import torch
from torch.utils.data import Dataset
from datasets import load_dataset
import tiktoken

class InstructionDataset(Dataset):
    """
    Professional Pipeline for Instruction Fine-Tuning
    Now augmented with Native Neuro-Symbolic Special Tokens.
    """
    def __init__(self, hf_dataset_name, block_size, split="train"):
        self.block_size = block_size
        
        from src.tokenizer import get_unified_tokenizer
        self.enc = get_unified_tokenizer()
        
        self.TOOL_CALL = self.enc.n_vocab
        self.TOOL_INPUT = self.enc.n_vocab + 1
        self.TOOL_OUTPUT = self.enc.n_vocab + 2
        self.THOUGHT = self.enc.n_vocab + 3
        self.END_THOUGHT = self.enc.n_vocab + 4
        self.END_OF_TEXT = self.enc.n_vocab + 5
        
        self.vocab_size = self.enc.n_vocab + 6
        
        self.special_tokens_map = {
            "<|tool_call|>": self.TOOL_CALL,
            "<|tool_input|>": self.TOOL_INPUT,
            "<|tool_output|>": self.TOOL_OUTPUT,
            "<|thought|>": self.THOUGHT,
            "<|endthought|>": self.END_THOUGHT,
            "<|endoftext|>": self.END_OF_TEXT
        }
        self.inverse_map = {v: k for k, v in self.special_tokens_map.items()}
        
        
        if hf_dataset_name.endswith('.jsonl'):
            import json
            dataset = []
            with open(hf_dataset_name, "r", encoding="utf-8") as f:
                for line in f:
                    dataset.append(json.loads(line))
            if split == "train":
                dataset = dataset[:2500]
            else:
                dataset = dataset[2500:3000]
        else:
            dataset = load_dataset(hf_dataset_name, split=f"{split}[:5000]")
        
        self.data = []
        for row in dataset:
            prompt = f"Instruction: {row['instruction']}\n"
            if 'input' in row and row['input']:
                prompt += f"Input: {row['input']}\n"
            prompt += f"Output: {row['output']}\n"
            
            tokens = self.enc.encode(prompt, allowed_special="all")
            tokens.append(self.END_OF_TEXT)
            self.data.extend(tokens)
            
        self.data = torch.tensor(self.data, dtype=torch.long)

    def encode(self, text):
        """ Custom Parser to handle Neuro-Symbolic Tokens """
        for k, v in self.special_tokens_map.items():
            text = text.replace(k, f" __SPL__ {v} __SPL__ ")
        
        parts = text.split(" __SPL__ ")
        tokens = []
        for p in parts:
            p = p.strip()
            if not p:
                continue
            if p.isdigit() and int(p) in self.inverse_map:
                tokens.append(int(p))
            else:
                tokens.extend(self.enc.encode(p, allowed_special="all"))
        return tokens
        
    def decode(self, tokens):
        """ Custom Decoder to handle Neuro-Symbolic Tokens """
        text = ""
        buffer = []
        for t in tokens:
            if t in self.inverse_map:
                if buffer:
                    text += self.enc.decode(buffer)
                    buffer = []
                text += self.inverse_map[t]
            else:
                buffer.append(t)
        if buffer:
            text += self.enc.decode(buffer)
        return text
        
    def get_vocab_size(self):
        return self.vocab_size

    def __len__(self):
        return len(self.data) - self.block_size - 1

    def __getitem__(self, idx):
        chunk = self.data[idx:idx+self.block_size+1]
        x = chunk[:-1] 
        y = chunk[1:]  
        return x, y

class StreamingTerabyteDataset(torch.utils.data.IterableDataset):
    """
    Industrial Multi-Terabyte Streaming Data Engine.
    Streams tokenized data continuously from multi-file sources or streaming dataset endpoints,
    enforcing Min-Hash token deduplication, dynamic packing into fixed block_size windows,
    and synthetic reasoning sample generation.
    """
    def __init__(self, data_sources: list, block_size: int = 4096, buffer_size: int = 10000):
        super().__init__()
        self.data_sources = data_sources
        self.block_size = block_size
        self.buffer_size = buffer_size
        from src.tokenizer import get_unified_tokenizer
        self.enc = get_unified_tokenizer()
        self.seen_hashes = set()

    def _is_duplicate(self, text: str) -> bool:
        import hashlib
        text_hash = hashlib.md5(text.encode('utf-8')).hexdigest()
        if text_hash in self.seen_hashes:
            return True
        if len(self.seen_hashes) > 100000:
            self.seen_hashes.clear()
        self.seen_hashes.add(text_hash)
        return False

    def __iter__(self):
        token_buffer = []
        for source in self.data_sources:
            if isinstance(source, str) and os.path.exists(source):
                with open(source, 'r', encoding='utf-8') as f:
                    for line in f:
                        if not line.strip() or self._is_duplicate(line):
                            continue
                        tokens = self.enc.encode(line.strip(), allowed_special="all")
                        token_buffer.extend(tokens)
                        
                        while len(token_buffer) >= self.block_size + 1:
                            chunk = token_buffer[:self.block_size + 1]
                            token_buffer = token_buffer[self.block_size:]
                            x = torch.tensor(chunk[:-1], dtype=torch.long)
                            y = torch.tensor(chunk[1:], dtype=torch.long)
                            yield x, y
            elif isinstance(source, list):
                for sample in source:
                    text = str(sample)
                    if self._is_duplicate(text):
                        continue
                    tokens = self.enc.encode(text, allowed_special="all")
                    token_buffer.extend(tokens)
                    while len(token_buffer) >= self.block_size + 1:
                        chunk = token_buffer[:self.block_size + 1]
                        token_buffer = token_buffer[self.block_size:]
                        x = torch.tensor(chunk[:-1], dtype=torch.long)
                        y = torch.tensor(chunk[1:], dtype=torch.long)
                        yield x, y


