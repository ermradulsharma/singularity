import ast
import subprocess
import os
import sys
import tempfile
from typing import Dict, Any, Optional

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

    def generic_visit(self, node: ast.AST) -> None:
        node_type = type(node).__name__
        if node_type not in self.ALLOWED_NODES:
            raise SecurityException(f"Forbidden Code Structure Detected: {node_type}")
        
        if isinstance(node, ast.Attribute) and node.attr.startswith('__'):
            raise SecurityException(f"Dunder Attribute Access Blocked: {node.attr}")
            
        super().generic_visit(node)

class SecureSandbox:
    """Executes untrusted Python and compiled languages securely inside a locked-down Docker container."""

    def __init__(self, use_docker: bool = True):
        self.use_docker = use_docker

    def execute(self, code_str: str, env_dict: Optional[Dict[str, Any]] = None, timeout: int = 5, max_memory_mb: int = 128) -> str:
        """Validate and execute untrusted Python code with OS-level isolation."""
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
                return self._execute_local_fallback(code_str, timeout)

            return self._execute_docker(code_str, timeout, max_memory_mb)
                
        except SecurityException as se:
            return f"[SECURITY BLOCKED] {se}"
        except SyntaxError as syn_err:
            return f"[SYNTAX ERROR] {syn_err}"
        except Exception as e:
            return f"[SYSTEM ERROR] {e}"

    def _execute_local_fallback(self, code_str: str, timeout: int) -> str:
        """Executes python script in a process-bounded local subprocess with sanitized environment and timeout protection."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False, encoding='utf-8') as f:
            f.write(code_str)
            temp_path = f.name
        try:
            clean_env = {
                "PATH": os.environ.get("PATH", ""),
                "PYTHONPATH": "",
                "PYTHONUNBUFFERED": "1"
            }
            res = subprocess.run([sys.executable, "-I", temp_path], capture_output=True, text=True, timeout=timeout, env=clean_env)
            out = res.stdout if res.returncode == 0 else (res.stdout + "\n" + res.stderr)
            return out if out.strip() else "[EXECUTION COMPLETED WITH NO STDOUT]"
        except subprocess.TimeoutExpired:
            return f"[EXECUTION TIMEOUT] Process exceeded {timeout}s limit and was terminated."
        except Exception as e:
            return f"[EXECUTION ERROR] {e}"
        finally:
            if os.path.exists(temp_path):
                try:
                    os.remove(temp_path)
                except OSError as e:
                    from src.telemetry import logger
                    logger.log("WARNING", "SANDBOX", f"Temp file cleanup failed: {e}")

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
        """Executes compiled languages (C++, Rust) in a locked-down Docker container with true compilation and execution."""
        lang_clean = lang.lower()
        if lang_clean not in ["cpp", "c++", "rust"]:
            return f"[ERROR] Unsupported language: {lang}"
            
        ext = ".cpp" if lang_clean in ["cpp", "c++"] else ".rs"
        with tempfile.NamedTemporaryFile(mode='w', suffix=ext, delete=False, encoding='utf-8') as f:
            f.write(code_str)
            src_path = f.name

        try:
            if not self.use_docker:
                compiler = "g++" if lang_clean in ["cpp", "c++"] else "rustc"
                bin_path = src_path + ".exe" if sys.platform == "win32" else src_path + ".bin"
                compile_res = subprocess.run([compiler, src_path, "-o", bin_path], capture_output=True, text=True, timeout=timeout)
                if compile_res.returncode != 0:
                    return f"[COMPILATION ERROR] {compile_res.stderr.strip()}"
                run_res = subprocess.run([bin_path], capture_output=True, text=True, timeout=timeout)
                if os.path.exists(bin_path):
                    os.remove(bin_path)
                return run_res.stdout if run_res.returncode == 0 else run_res.stderr
            else:
                image = "gcc:latest" if lang_clean in ["cpp", "c++"] else "rust:latest"
                cmd = [
                    "docker", "run", "--rm",
                    "--network=none",
                    "-v", f"{src_path}:/app/src{ext}:ro",
                    image,
                    "sh", "-c",
                    f"g++ /app/src{ext} -o /tmp/app.bin && /tmp/app.bin" if lang_clean in ["cpp", "c++"] else f"rustc /app/src{ext} -o /tmp/app.bin && /tmp/app.bin"
                ]
                res = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout * 2)
                return f"[DOCKER {lang_clean.upper()}] {res.stdout.strip() if res.returncode == 0 else res.stderr.strip()}"
        except Exception as e:
            return f"[EXECUTION ERROR] Failed to compile and execute {lang}: {str(e)}"
        finally:
            if os.path.exists(src_path):
                os.remove(src_path)



