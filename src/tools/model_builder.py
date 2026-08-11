import os
import ast

def generate_sub_brain(model_name: str, pytorch_code: str) -> str:
    """Autonomously generates and saves a SubBrain neural adapter to the AGI's architecture. The code must contain a 'class SubBrain(nn.Module):' definition."""
    # 1. Validation
    if not model_name.isidentifier():
        return "[ERROR] Invalid model_name. Use snake_case, e.g. 'gemini_style'."
    
    try:
        # Check for syntax errors and the required SubBrain class
        tree = ast.parse(pytorch_code)
        has_subbrain_class = any(isinstance(node, ast.ClassDef) and node.name == "SubBrain" for node in tree.body)
        if not has_subbrain_class:
            return "[ERROR] The provided PyTorch code must contain a 'class SubBrain(nn.Module):' definition."
    except SyntaxError as e:
        return f"[ERROR] Syntax error in the generated PyTorch code: {e}"

    # 2. Save location
    sub_brains_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "sub_brains")
    os.makedirs(sub_brains_dir, exist_ok=True)
    
    filepath = os.path.join(sub_brains_dir, f"{model_name}.py")
    
    # 3. Write to disk (Self-Modification)
    try:
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(pytorch_code)
        return f"[SUCCESS] Sub-Brain '{model_name}' successfully generated and injected into {filepath}. It will be absorbed upon next boot."
    except Exception as e:
        return f"[ERROR] Failed to save Sub-Brain: {e}"
