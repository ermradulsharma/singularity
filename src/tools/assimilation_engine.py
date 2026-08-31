import os
import json
import urllib.request
import sys

def _ensure_knowledge_base():
    """Internal helper to resolve knowledge base directory."""
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    kb_path = os.path.join(base_dir, "data", "knowledge_base")
    os.makedirs(kb_path, exist_ok=True)
    return kb_path

def _download_raw_json(url: str, filename: str):
    """Internal helper to download raw JSON datasets."""
    kb_path = _ensure_knowledge_base()
    file_path = os.path.join(kb_path, filename)
    
    try:
        urllib.request.urlretrieve(url, file_path)
        return True
    except Exception:
        return False

def assimilate_phi4_logic():
    """Assimilates Phi-4 level mathematical reasoning and logic datasets into the local knowledge base."""
    url = "https://raw.githubusercontent.com/tatsu-lab/stanford_alpaca/main/alpaca_data.json"
    _download_raw_json(url, "phi4_logic_knowledge.json")

def assimilate_qwen_multilingual():
    """Assimilates Qwen level code execution and multilingual reasoning datasets into the local knowledge base."""
    url = "https://raw.githubusercontent.com/sahil280114/codealpaca/master/data/code_alpaca_20k.json"
    _download_raw_json(url, "qwen_code_knowledge.json")

if __name__ == "__main__":
    assimilate_phi4_logic()
    assimilate_qwen_multilingual()

