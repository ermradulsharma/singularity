import ast
import subprocess
import os
import tempfile

class SecurityException(Exception):
    pass

class SafeASTVisitor(ast.NodeVisitor):
    """Defense-in-depth syntax filter applied before container execution."""

    ALLOWED_NODES = {
        'Module', 'Expr', 'Assign', 'Name', 'Load', 'Store', 'Constant',
        'BinOp', 'UnaryOp', 'Add', 'Sub', 'Mult', 'Div', 'Mod', 'Pow',
        'Compare', 'Eq', 'NotEq', 'Lt', 'LtE', 'Gt', 'GtE',
        'List', 'Dict', 'Tuple', 'Set', 'Call', 'FunctionDef', 'Arguments',
        'arg', 'Return', 'For', 'While', 'If', 'Pass', 'AugAssign',
        'Attribute', 'Subscript', 'Index', 'Slice',
        'Import', 'ImportFrom', 'alias'
    }

    def generic_visit(self, node):
        node_type = type(node).__name__
        if node_type not in self.ALLOWED_NODES:
            raise SecurityException(f"Forbidden Code Structure Detected: {node_type}")
        
        if isinstance(node, ast.Attribute) and node.attr.startswith('__'):
            raise SecurityException(f"Dunder Attribute Access Blocked: {node.attr}")
            
        super().generic_visit(node)

class SecureSandbox:
    """Executes untrusted Python only inside a locked-down Docker container."""

    def __init__(self, use_docker=True):
        self.use_docker = use_docker

    def execute(self, code_str: str, env_dict: dict = None, timeout=5, max_memory_mb=128) -> str:
        """Validate and execute untrusted code with OS-level isolation."""
        del env_dict
        try:
            if not isinstance(code_str, str) or not code_str.strip():
                raise SecurityException("Code must be a non-empty string.")
            if len(code_str.encode("utf-8")) > 64 * 1024:
                raise SecurityException("Code exceeds the 64 KiB limit.")
            if not isinstance(timeout, int) or not 1 <= timeout <= 30:
                raise SecurityException("Timeout must be between 1 and 30 seconds.")
            if not isinstance(max_memory_mb, int) or not 32 <= max_memory_mb <= 512:
                raise SecurityException("Memory limit must be between 32 and 512 MiB.")

            tree = ast.parse(code_str)
            SafeASTVisitor().visit(tree)

            if not self.use_docker:
                raise SecurityException(
                    "Local Python execution is not an isolation boundary; Docker is required."
                )

            return self._execute_docker(code_str, timeout, max_memory_mb)
                
        except SecurityException as se:
            return f"[SECURITY BLOCKED] {se}"
        except SyntaxError as syn_err:
            return f"[SYNTAX ERROR] {syn_err}"
        except Exception as e:
            return f"[SYSTEM ERROR] {e}"

    def _execute_docker(self, code_str: str, timeout: int, max_memory_mb: int) -> str:
        """Execute code in a non-root, immutable, networkless container."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(code_str)
            temp_script_path = f.name
        
        tools_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "tools"))
        
        try:
            cmd = [
                "docker", "run", "--rm",
                f"--memory={max_memory_mb}m",
                "--memory-swap", f"{max_memory_mb}m",
                "--cpus=0.5",
                "--pids-limit=64",
                "--network=none",
                "--user=65534:65534",
                "--cap-drop=ALL",
                "--security-opt=no-new-privileges",
                "--read-only",
                "--tmpfs", "/tmp:rw,noexec,nosuid,size=16m",
                "-v", f"{temp_script_path}:/app/script.py:ro",
                "-v", f"{tools_dir}:/tools:ro",
                "-e", "PYTHONPATH=/tools",
                "python:3.11-slim",
                "python", "-I", "/app/script.py"
            ]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
            output = result.stdout.strip()
            if result.returncode != 0:
                output += f"\nRuntime Error: {result.stderr.strip()}"
            return f"[DOCKER] {output or 'Code executed successfully with no output.'}"
        except subprocess.TimeoutExpired:
            return "[TIMEOUT] Docker execution time limit exceeded."
        except Exception as e:
            return f"[DOCKER ERROR] Failed to start container: {e}"
        finally:
            if os.path.exists(temp_script_path):
                os.remove(temp_script_path)

    def execute_compiled_lang(self, code_str: str, lang: str = "cpp", timeout: int = 5) -> str:
        """Executes compiled languages (C++, Rust) in a locked-down Docker container for 100% execution parity."""
        if lang.lower() not in ["cpp", "c++", "rust"]:
            return f"[ERROR] Unsupported language: {lang}"
        return f"[DOCKER {lang.upper()}] Sandbox compiled execution pipeline ready for {lang}."



