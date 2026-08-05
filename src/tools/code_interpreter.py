import os
import sys

# Attempt to import the sandbox
try:
    from src.sandbox import SecureSandbox
except ImportError:
    SecureSandbox = None

def run_python_code(code_str: str, use_docker: bool = True) -> str:
    """
    Executes a block of Python code securely inside the Swarm's Sandbox.
    Useful for testing algorithms, running complex calculations, or data parsing.
    """
    if not SecureSandbox:
        return "[SYSTEM ERROR] SecureSandbox module not found in src.sandbox."
        
    try:
        sandbox = SecureSandbox(use_docker=use_docker)
        # Execute the untrusted code
        result = sandbox.execute(code_str=code_str, timeout=10)
        return result
    except Exception as e:
        return f"[TOOL ERROR] Code Interpreter failed: {str(e)}"

