import json
import time
import os
import sys

class CognitiveTelemetry:
    """
    Strict AGI logger that writes structured JSONL events instead of unstructured prints.
    """
    def __init__(self, log_dir="data"):
        self.log_dir = log_dir
        os.makedirs(self.log_dir, exist_ok=True)
        self.log_file = os.path.join(self.log_dir, "telemetry.jsonl")

    def log(self, level: str, module: str, message: str, **kwargs):
        """Logs a structured JSON event."""
        event = {
            "timestamp": time.time(),
            "level": level.upper(),
            "module": module,
            "message": message,
            **kwargs
        }
        
        try:
            with open(self.log_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(event) + "\n")
        except Exception as e:
            pass
            
        try:
            sys.stdout.write(f"[{level.upper()}] [{module}] {message}\n")
            sys.stdout.flush()
        except (UnicodeEncodeError, AttributeError):
            clean_msg = message.encode(getattr(sys.stdout, 'encoding', 'utf-8') or 'utf-8', errors='replace').decode(getattr(sys.stdout, 'encoding', 'utf-8') or 'utf-8')
            sys.stdout.write(f"[{level.upper()}] [{module}] {clean_msg}\n")
            sys.stdout.flush()

logger = CognitiveTelemetry()

