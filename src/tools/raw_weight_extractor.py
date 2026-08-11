from src.telemetry import logger
import os
import sys

def check_dependencies():
    try:
        import huggingface_hub
    except ImportError:
        print("[!] Missing dependency. Please run: pip install huggingface_hub")
        sys.exit(1)

def extract_raw_weights(model_id: str):
    """Autonomously downloads the RAW weights, configs, and datasets for a given HuggingFace model. NO architecture translation is performed. The AGI will absorb this data internally later."""
    from huggingface_hub import snapshot_download
    
    # Safe directory name based on model_id
    model_dir_name = model_id.replace("/", "_")
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    save_path = os.path.join(base_dir, "data", "raw_weights", model_dir_name)
    
    print(f"\n[*] Target: {model_id}")
    print(f"[*] Extracting raw knowledge to: {save_path}")
    print("[*] Note: This may take a long time depending on the model size (e.g. 15GB+).")
    
    try:
        # Download all safetensors, bin, json, and tokenizer files
        snapshot_download(
            repo_id=model_id,
            local_dir=save_path,
            local_dir_use_symlinks=False,
            resume_download=True,
            ignore_patterns=["*.msgpack", "*.h5", "*.ot", "coreml/*", "onnx/*"]
        )
        print(f"\n[SUCCESS] Raw Data Extracted Successfully: {save_path}")
        print("[SYSTEM] The Master AGI can now begin Autonomous Assimilation on these weights.")
        
    except Exception as e:
        print(f"\n[ERROR] Failed to extract data: {e}")

if __name__ == "__main__":
    check_dependencies()
    print("============================================================")
    print("[*] AGI AUTONOMOUS KNOWLEDGE EXTRACTOR (Raw Tensors & Data) [*]")
    print("============================================================")
    print("1. Extract Phi-4-mini (3.8B) - Math & Logic")
    print("2. Extract Qwen3-4B / Qwen2.5-3B - Multilingual & Coding")
    print("3. Extract Llama 3.2 (3B) - Broad Knowledge")
    print("4. Extract Gemma 3 / Gemma 2 (2B) - Google's Multimodal")
    print("5. Extract Custom Model (Enter HuggingFace ID)")
    
    choice = input("Enter choice (1-5): ")
    
    if choice == '1':
        extract_raw_weights("microsoft/Phi-3-mini-4k-instruct") # Nearest equivalent for testing
    elif choice == '2':
        extract_raw_weights("Qwen/Qwen2.5-3B-Instruct")
    elif choice == '3':
        extract_raw_weights("meta-llama/Llama-3.2-3B-Instruct")
    elif choice == '4':
        extract_raw_weights("google/gemma-2-2b-it")
    elif choice == '5':
        custom_id = input("Enter HuggingFace Model ID (e.g. HuggingFaceTB/SmolLM-135M): ")
        extract_raw_weights(custom_id.strip())
    else:
        print("Invalid choice. Exiting.")
