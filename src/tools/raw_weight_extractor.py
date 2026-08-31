import os
import sys

def _check_dependencies():
    """Internal helper to ensure huggingface_hub is available."""
    try:
        import huggingface_hub
    except ImportError:
        sys.exit(1)

def extract_raw_weights(model_id: str):
    """Autonomously downloads raw weights, configs, and datasets from HuggingFace without architecture translation."""
    from huggingface_hub import snapshot_download
    
    model_dir_name = model_id.replace("/", "_")
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    save_path = os.path.join(base_dir, "data", "raw_weights", model_dir_name)
    
    try:
        download_path = snapshot_download(
            repo_id=model_id,
            local_dir=save_path,
            local_dir_use_symlinks=False,
            resume_download=True,
            ignore_patterns=["*.msgpack", "*.h5", "*.ot", "coreml/*", "onnx/*"]
        )
        return {"status": "success", "message": f"Successfully extracted weights to {download_path}"}
    except Exception as e:
        return {"status": "error", "message": f"Failed to extract raw weights for '{model_id}': {str(e)}"}

if __name__ == "__main__":
    _check_dependencies()
    if len(sys.argv) > 1:
        custom_id = sys.argv[1]
        if custom_id and custom_id.strip():
            extract_raw_weights(custom_id.strip())

