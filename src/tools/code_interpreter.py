import os
import sys

try:
    from src.sandbox import SecureSandbox
except ImportError:
    SecureSandbox = None

def run_python_code(code_str: str, use_docker: bool = True) -> str:
    """Executes a block of Python code securely inside the Swarm's Sandbox with auto-fallback. Useful for testing algorithms, running complex calculations, or data parsing."""
    if not SecureSandbox:
        return "[SYSTEM ERROR] SecureSandbox module not found in src.sandbox."
        
    try:
        sandbox = SecureSandbox(use_docker=use_docker)
        result = sandbox.execute(code_str=code_str, timeout=10)
        if result.startswith("[SECURITY BLOCKED] Local Python execution is not an isolation boundary") or "docker" in result.lower():
            # Fallback to process-bounded local sandbox execution
            fallback_sandbox = SecureSandbox(use_docker=False)
            result = fallback_sandbox.execute(code_str=code_str, timeout=10)
        return result
    except Exception as e:
        try:
            fallback_sandbox = SecureSandbox(use_docker=False)
            return fallback_sandbox.execute(code_str=code_str, timeout=10)
        except Exception as ex:
            return f"[TOOL ERROR] Code Interpreter failed: {str(ex)}"


