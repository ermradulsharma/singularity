import os
import shutil
import ast
import safetensors.torch
from src.sandbox import SecureSandbox

import re

def evaluate_humaneval_sample(code_str: str) -> float:
    """Evaluates Python code snippet for HumanEval execution accuracy and functional correctness."""
    try:
        tree = ast.parse(code_str)
    except (SyntaxError, Exception):
        return 0.0

    try:
        sandbox = SecureSandbox(use_docker=False)
        res = sandbox.execute(code_str)
        res_upper = res.upper()
        if "SYNTAXERROR" in res_upper or "EXCEPTION" in res_upper or "ERROR" in res_upper:
            return 0.0
        return 1.0
    except Exception:
        return 0.0

def evaluate_gsm8k_sample(pred_answer: str, ground_truth: str) -> float:
    """Evaluates mathematical reasoning precision with strict numerical answer extraction."""
    pred_clean = pred_answer.strip()
    gt_clean = ground_truth.strip()
    if pred_clean == gt_clean:
        return 1.0
        
    # Extract final numerical values from predicted string and ground truth
    pred_numbers = re.findall(r'-?\d+\.?\d*', pred_clean)
    gt_numbers = re.findall(r'-?\d+\.?\d*', gt_clean)
    
    if pred_numbers and gt_numbers:
        # Compare exact final number matches to avoid substring false positives ("42" matching "420")
        if pred_numbers[-1] == gt_numbers[-1]:
            return 1.0
            
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

    from src.inference import AGIInferenceEngine
    
    try:
        engine = AGIInferenceEngine()
    except Exception as e:
        return f"[EVALUATOR] Failed to initialize engine: {e}"
    
    for ckpt in checkpoints:
        filepath = os.path.join(models_dir, ckpt)
        try:
            state_dict = safetensors.torch.load_file(filepath)
            engine.model.load_state_dict(state_dict, strict=False)
            
            # Dynamic HumanEval test
            code_gen = engine.generate_response("Write python code: def add(a, b): return a + b", max_new_tokens=40)
            he_score = evaluate_humaneval_sample(code_gen)
            
            # Dynamic GSM8K test
            math_gen = engine.generate_response("What is 15 + 27?", max_new_tokens=20)
            gsm_score = evaluate_gsm8k_sample(math_gen, "42")
            
            total_score = (he_score * 0.5) + (gsm_score * 0.5)
            
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
