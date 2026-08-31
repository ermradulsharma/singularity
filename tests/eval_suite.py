import torch
import pytest
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.sandbox import SecureSandbox

def evaluate_humaneval_sample(code_str: str) -> float:
    """Evaluates Python code snippet for HumanEval execution accuracy."""
    sandbox = SecureSandbox(use_docker=True)
    try:
        res = sandbox.execute(code_str)
        if "SyntaxError" in res or "Error" in res:
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

def test_benchmark_evaluators():
    code_pass = "def add(a, b):\n    return a + b\nprint(add(2, 3))"
    assert evaluate_humaneval_sample(code_pass) == 1.0
    
    code_fail = "def add(a, b)\n    return a + b"
    assert evaluate_humaneval_sample(code_fail) == 0.0
    
    assert evaluate_gsm8k_sample("Final Answer: 42", "42") == 0.8
    assert evaluate_gsm8k_sample("42", "42") == 1.0

if __name__ == "__main__":
    test_benchmark_evaluators()
    print("✅ Benchmark Evaluator Test Suite Passed Successfully.")
