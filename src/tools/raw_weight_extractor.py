from src.telemetry import logger
import os
import sys

def _check_dependencies():
    """Internal helper to ensure huggingface_hub is available."""
    try:
        import huggingface_hub
    except ImportError:
        print("[!] Missing dependency. Please run: pip install huggingface_hub")
        sys.exit(1)

def extract_raw_weights(model_id: str):
    """Autonomously downloads raw weights, configs, and datasets from HuggingFace without architecture translation."""
    from huggingface_hub import snapshot_download
    
    model_dir_name = model_id.replace("/", "_")
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    save_path = os.path.join(base_dir, "data", "raw_weights", model_dir_name)
    
    print(f"\n[*] Target: {model_id}")
    print(f"[*] Extracting raw knowledge to: {save_path}")
    print("[*] Note: This may take a long time depending on the model size (e.g. 15GB+).")
    
    try:
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
    _check_dependencies()
    print("============================================================")
    print("[*] AGI AUTONOMOUS KNOWLEDGE EXTRACTOR (Raw Tensors & Data) [*]")
    print("============================================================")
    
    if len(sys.argv) > 1:
        custom_id = sys.argv[1]
    else:
        custom_id = input("Enter HuggingFace Model ID (e.g. Qwen/Qwen2.5-1.5B): ")
        
    if custom_id and custom_id.strip():
        extract_raw_weights(custom_id.strip())
    else:
        print("[ERROR] No Model ID provided. Exiting.")

