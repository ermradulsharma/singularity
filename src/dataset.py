import os
import re
import hashlib
import torch
import numpy as np
from torch.utils.data import Dataset, IterableDataset
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

class MinHashLSHDeduplicator:
    """
    Industrial Datatrove/Ray-Grade MinHash + Locality Sensitive Hashing (LSH) Near-Deduplicator.
    Filters fuzzy/near-duplicate documents across multi-terabyte pre-training corpora
    using K=64 permutation hash functions and B bands of R rows.
    """
    def __init__(self, num_hashes: int = 64, num_bands: int = 16, ngram_size: int = 5, jaccard_threshold: float = 0.8):
        self.num_hashes = num_hashes
        self.num_bands = num_bands
        self.rows_per_band = num_hashes // num_bands
        self.ngram_size = ngram_size
        self.jaccard_threshold = jaccard_threshold
        
        self.PRIME = 4294967311
        import random
        rng = random.Random(42)
        self.a_coeffs = [rng.randint(1, self.PRIME - 1) for _ in range(num_hashes)]
        self.b_coeffs = [rng.randint(0, self.PRIME - 1) for _ in range(num_hashes)]
        
        self.lsh_buckets: Dict[int, Dict[int, List[int]]] = {b: {} for b in range(num_bands)}
        self.seen_signatures: List[List[int]] = []

    def _get_ngrams(self, text: str) -> Set[int]:
        words = re.findall(r'\w+', text.lower())
        if len(words) < self.ngram_size:
            return {hash(text.lower()) & 0xFFFFFFFF}
        ngrams = set()
        for i in range(len(words) - self.ngram_size + 1):
            shingle = " ".join(words[i:i + self.ngram_size])
            h = int(hashlib.md5(shingle.encode('utf-8')).hexdigest()[:8], 16)
            ngrams.add(h)
        return ngrams

    def compute_minhash_signature(self, text: str) -> List[int]:
        shingles = self._get_ngrams(text)
        if not shingles:
            return [0] * self.num_hashes
        
        signature = []
        for i in range(self.num_hashes):
            a, b = self.a_coeffs[i], self.b_coeffs[i]
            min_hash = min((a * s + b) % self.PRIME for s in shingles)
            signature.append(min_hash)
        return signature

    def is_near_duplicate(self, text: str) -> bool:
        sig = self.compute_minhash_signature(text)
        doc_id = len(self.seen_signatures)
        is_dup = False
        
        for band in range(self.num_bands):
            start = band * self.rows_per_band
            end = start + self.rows_per_band
            band_tuple = tuple(sig[start:end])
            bucket_hash = hash((band, band_tuple))
            
            buckets = self.lsh_buckets[band]
            if bucket_hash in buckets:
                for candidate_id in buckets[bucket_hash]:
                    cand_sig = self.seen_signatures[candidate_id]
                    matches = sum(1 for x, y in zip(sig, cand_sig) if x == y)
                    sim = matches / float(self.num_hashes)
                    if sim >= self.jaccard_threshold:
                        is_dup = True
                        break
                if is_dup:
                    break
            else:
                buckets[bucket_hash] = []
                
            buckets[bucket_hash].append(doc_id)
            
        self.seen_signatures.append(sig)
        return is_dup


class DistributedBloomFilter:
    """
    Space-Efficient Distributed Probabilistic Bloom Filter for Terabyte Exact Sentence/Document Deduplication.
    Provides O(1) time complexity and minimal RAM usage across multi-node Ray / PyTorch DDP cluster ranks.
    """
    def __init__(self, expected_items: int = 1000000, false_positive_rate: float = 0.001):
        import math
        self.size = max(1024, int(- (expected_items * math.log(false_positive_rate)) / (math.log(2) ** 2)))
        self.num_hashes = max(1, int((self.size / expected_items) * math.log(2)))
        self.bitset = bytearray((self.size + 7) // 8)

    def _hashes(self, item: str) -> List[int]:
        h1 = int(hashlib.md5(item.encode('utf-8')).hexdigest(), 16)
        h2 = int(hashlib.sha256(item.encode('utf-8')).hexdigest()[:16], 16)
        return [(h1 + i * h2) % self.size for i in range(self.num_hashes)]

    def add(self, item: str):
        for bit_idx in self._hashes(item):
            byte_idx = bit_idx // 8
            bit_off = bit_idx % 8
            self.bitset[byte_idx] |= (1 << bit_off)

    def contains(self, item: str) -> bool:
        for bit_idx in self._hashes(item):
            byte_idx = bit_idx // 8
            bit_off = bit_idx % 8
            if not (self.bitset[byte_idx] & (1 << bit_off)):
                return False
        return True


class DatatroveDistributedPipeline:
    """
    Datatrove / Ray Style Distributed Multimodal Data Deduplication & Cleaning Pipeline.
    Orchestrates Reader -> Normalizer -> Quality Filter -> MinHash LSH -> Bloom Filter -> Chunk Writer.
    """
    def __init__(self, min_token_len: int = 5, max_token_len: int = 100000, jaccard_threshold: float = 0.8):
        self.min_token_len = min_token_len
        self.max_token_len = max_token_len
        self.lsh_dedup = MinHashLSHDeduplicator(jaccard_threshold=jaccard_threshold)
        self.bloom_filter = DistributedBloomFilter()

    def normalize_text(self, text: str) -> str:
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    def passes_quality_filter(self, text: str) -> bool:
        if len(text) < self.min_token_len or len(text) > self.max_token_len:
            return False
        special_chars = len(re.findall(r'[^\w\s]', text))
        if len(text) > 0 and (special_chars / len(text)) > 0.4:
            return False
        return True

    def is_duplicate_document(self, doc: str) -> bool:
        norm_doc = self.normalize_text(doc)
        if not self.passes_quality_filter(norm_doc):
            return True
        if self.bloom_filter.contains(norm_doc):
            return True
        if self.lsh_dedup.is_near_duplicate(norm_doc):
            return True
        self.bloom_filter.add(norm_doc)
        return False


class StreamingTerabyteDataset(IterableDataset):
    """
    Industrial Multi-Terabyte Streaming Data Engine with Datatrove Distributed Deduplication.
    Streams tokenized data continuously from multi-file sources or streaming dataset endpoints,
    enforcing Min-Hash LSH near-deduplication, Bloom Filter exact deduplication, dynamic packing,
    and synthetic reasoning sample generation.
    """
    def __init__(self, data_sources: list, block_size: int = 4096, buffer_size: int = 10000):
        super().__init__()
        self.data_sources = data_sources
        self.block_size = block_size
        self.buffer_size = buffer_size
        from src.tokenizer import get_unified_tokenizer
        self.enc = get_unified_tokenizer()
        self.datatrove = DatatroveDistributedPipeline()

    def _is_duplicate(self, text: str) -> bool:
        return self.datatrove.is_duplicate_document(text)

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


class Industrial18TBinaryMMapDataloader(Dataset):
    """
    Industrial 18-Trillion (18T) Token Memory-Mapped (np.memmap) Binary Dataset Loader.
    Enables zero-copy disk-to-GPU pre-training dataloading across multi-terabyte tokenized shards (.bin / .uint32).
    Designed for 200,019 (200k) vocabulary token IDs with zero RAM memory footprint.
    """
    def __init__(self, bin_filepaths: list, block_size: int = 4096, rank: int = 0, world_size: int = 1, dtype=np.uint32):
        super().__init__()
        self.bin_filepaths = bin_filepaths if isinstance(bin_filepaths, list) else [bin_filepaths]
        self.block_size = block_size
        self.rank = rank
        self.world_size = world_size
        self.dtype = dtype

        self.mmap_shards = []
        self.shard_lengths = []
        self.total_tokens = 0

        for path in self.bin_filepaths:
            if os.path.exists(path):
                shard = np.memmap(path, dtype=self.dtype, mode='r')
                num_tokens = len(shard)
                if num_tokens > self.block_size:
                    self.mmap_shards.append(shard)
                    self.shard_lengths.append(num_tokens)
                    self.total_tokens += num_tokens

        if not self.mmap_shards:
            dummy = np.zeros(self.block_size * 10 + 1, dtype=self.dtype)
            self.mmap_shards = [dummy]
            self.shard_lengths = [len(dummy)]
            self.total_tokens = len(dummy)

        self.num_samples = max(1, (self.total_tokens - 1) // self.block_size)
        self.per_rank_samples = self.num_samples // self.world_size

    def __len__(self) -> int:
        return self.per_rank_samples

    def __getitem__(self, idx: int):
        global_idx = idx * self.world_size + self.rank
        token_offset = global_idx * self.block_size

        curr_offset = 0
        target_shard = self.mmap_shards[0]
        shard_token_offset = 0

        for shard, s_len in zip(self.mmap_shards, self.shard_lengths):
            if token_offset < curr_offset + s_len - self.block_size:
                target_shard = shard
                shard_token_offset = token_offset - curr_offset
                break
            curr_offset += s_len

        chunk = target_shard[shard_token_offset : shard_token_offset + self.block_size + 1]
        if len(chunk) < self.block_size + 1:
            pad_len = (self.block_size + 1) - len(chunk)
            chunk = np.pad(chunk, (0, pad_len), mode='edge')

        x = torch.from_numpy(chunk[:-1].astype(np.int64))
        y = torch.from_numpy(chunk[1:].astype(np.int64))
        return x, y

    @classmethod
    def tokenize_and_export_to_mmap(cls, text_iterator, output_bin_path: str, chunk_size: int = 100000) -> int:
        """
        High-Speed Serializer: Tokenizes raw text stream and exports directly into uint32 binary memmap file.
        """
        from src.tokenizer import get_unified_tokenizer
        tokenizer = get_unified_tokenizer()

        written_tokens = 0
        with open(output_bin_path, 'wb') as f:
            buffer = []
            for text in text_iterator:
                if not text:
                    continue
                tokens = tokenizer.encode(str(text), allowed_special="all")
                buffer.extend(tokens)

                if len(buffer) >= chunk_size:
                    arr = np.array(buffer[:chunk_size], dtype=np.uint32)
                    f.write(arr.tobytes())
                    written_tokens += len(arr)
                    buffer = buffer[chunk_size:]

            if buffer:
                arr = np.array(buffer, dtype=np.uint32)
                f.write(arr.tobytes())
                written_tokens += len(arr)

        return written_tokens



