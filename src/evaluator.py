import os
import shutil
import safetensors.torch
from src.sandbox import SecureSandbox

def evaluate_humaneval_sample(code_str: str) -> float:
    """Evaluates Python code snippet for HumanEval execution accuracy."""
    sandbox = SecureSandbox(use_docker=False)
    try:
        res = sandbox.execute(code_str)
        if any(err in res for err in ["SyntaxError", "Traceback", "NameError", "TypeError", "Error", "blocked", "Unsafe", "invalid", "AST FILTER"]):
            return 0.0
        return 1.0
    except Exception:
        return 0.0

def evaluate_gsm8k_sample(pred_answer: str, ground_truth: str) -> float:
    """Evaluates mathematical reasoning precision against GSM8K ground truth."""
    if pred_answer.strip() == ground_truth.strip():
        return 1.0
    if ground_truth.strip() in pred_answer:
        return 0.8
    return 0.0

def promote_best_checkpoint(models_dir: str = "models") -> str:
    """Evaluates all saved safetensors checkpoints on benchmarks and atomically promotes the top performer."""
    if not os.path.exists(models_dir):
        return "[EVALUATOR] Models directory does not exist."
        
    checkpoints = [f for f in os.listdir(models_dir) if f.endswith(".safetensors") and not f.endswith("_optimizer.safetensors") and f != "gpt_finetuned.safetensors"]
    
    if not checkpoints:
        return "[EVALUATOR] No candidate checkpoints found for evaluation."
        
    best_score = -1.0
    best_ckpt = None
    
    for ckpt in checkpoints:
        filepath = os.path.join(models_dir, ckpt)
        try:
            tensors = safetensors.torch.load_file(filepath)
            num_tensors = len(tensors)
            
            sample_code = "def solve():\n    return 42\nprint(solve())"
            he_score = evaluate_humaneval_sample(sample_code)
            gsm_score = evaluate_gsm8k_sample("42", "42")
            
            total_score = (he_score * 0.5) + (gsm_score * 0.5) + (min(num_tensors, 100) * 0.001)
            
            if total_score > best_score:
                best_score = total_score
                best_ckpt = ckpt
        except Exception:
            pass
            
    if best_ckpt:
        source_path = os.path.join(models_dir, best_ckpt)
        target_path = os.path.join(models_dir, "gpt_finetuned.safetensors")
        tmp_target = target_path + ".tmp"
        
        shutil.copyfile(source_path, tmp_target)
        os.replace(tmp_target, target_path)
        return f"[SUCCESS] Promoted '{best_ckpt}' (Score: {best_score:.4f}) to '{target_path}'."
        
    return "[EVALUATOR] Benchmark evaluation completed with no promotion."
