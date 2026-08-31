import pytest
from src.dataset import MinHashLSHDeduplicator, DistributedBloomFilter, DatatroveDistributedPipeline, StreamingTerabyteDataset

def test_minhash_lsh_deduplicator():
    dedup = MinHashLSHDeduplicator(jaccard_threshold=0.8)
    doc1 = "The quick brown fox jumps over the lazy dog and runs through the forest."
    doc2 = "The quick brown fox jumps over the lazy dog and runs into the forest." # Near duplicate (~90% Jaccard)
    doc3 = "Quantum computing relies on quantum bits or qubits to perform complex matrix calculations." # Completely different
    
    assert not dedup.is_near_duplicate(doc1)
    assert dedup.is_near_duplicate(doc2)
    assert not dedup.is_near_duplicate(doc3)

def test_distributed_bloom_filter():
    bf = DistributedBloomFilter(expected_items=1000)
    sample = "Unique pre-training document content"
    assert not bf.contains(sample)
    bf.add(sample)
    assert bf.contains(sample)

def test_datatrove_pipeline():
    pipeline = DatatroveDistributedPipeline()
    doc1 = "DeepSeek-V3 Multi-Head Latent Attention compresses KV cache memory."
    doc2 = "DeepSeek-V3 Multi-Head Latent Attention compresses KV cache memory." # Exact duplicate
    
    assert not pipeline.is_duplicate_document(doc1)
    assert pipeline.is_duplicate_document(doc2)

def test_streaming_terabyte_dataset_with_dedup():
    sources = [
        "First document text for pre-training dataset.",
        "First document text for pre-training dataset.", # Duplicate line
        "Second document text with different content for pre-training."
    ]
    dataset = StreamingTerabyteDataset([sources], block_size=16)
    samples = list(dataset)
    assert len(samples) > 0
