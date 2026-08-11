import os
import ast

def create_future_tech_tool(technology_name: str, python_code: str) -> str:
    """FUTURE TECH SCIENTIST ENGINE: Generates a brand new logic engine (tool) for a new scientific domain or technology."""
    filename = f"{technology_name.lower().replace(' ', '_')}.py"
    tools_dir = os.path.dirname(os.path.abspath(__file__))
    filepath = os.path.join(tools_dir, filename)
    
    # 1. Syntax Check (Ensure the new science tool is valid logic)
    try:
        ast.parse(python_code)
    except SyntaxError as e:
        return f"Invention Failed: Syntax Error in {technology_name} logic - {e}"
        
    # 2. Save the Invention
    try:
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(python_code)
        return f"Invention Successful! {technology_name} tool created at {filepath}. It will be auto-discovered by the AGI brain instantly."
    except Exception as e:
        return f"Invention Failed: {e}"
