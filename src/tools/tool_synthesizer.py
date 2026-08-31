import os
import ast

def _validate_tool_syntax(code_str: str) -> bool:
    try:
        ast.parse(code_str)
        return True
    except SyntaxError:
        return False

def synthesize_new_tool(tool_name: str, python_code: str) -> str:
    """Autonomously synthesizes, verifies, and saves a new Python tool module to src/tools for live AST assimilation."""
    if not tool_name.isidentifier():
        return "[ERROR] Invalid tool_name. Use snake_case identifier."
        
    if not _validate_tool_syntax(python_code):
        return "[ERROR] Tool synthesis failed: Syntax error in PyTorch/Python code."
        
    tools_dir = os.path.join(os.path.dirname(__file__))
    filepath = os.path.join(tools_dir, f"{tool_name}.py")
    
    try:
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(python_code)
        return f"[SUCCESS] Tool '{tool_name}' successfully synthesized and saved to {filepath}. Ready for AST assimilation."
    except Exception as e:
        return f"[ERROR] Failed to save tool module: {str(e)}"
