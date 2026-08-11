import os
import ast

def rewrite_own_code(file_path: str, new_code: str) -> str:
    """META-EVOLUTION ENGINE: Allows the AGI to rewrite its own source code (e.g. src/model.py)."""
    if not os.path.exists(file_path):
        return f"Error: File {file_path} not found."
    
    # 1. Syntax Check (Safety mechanism to prevent brain-death)
    try:
        ast.parse(new_code)
    except SyntaxError as e:
        return f"Evolution Failed: Syntax Error in new code - {e}"
        
    # 2. Apply Evolution
    try:
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(new_code)
        return f"Evolution Successful. {file_path} has been biologically upgraded."
    except Exception as e:
        return f"Evolution Failed: {e}"
