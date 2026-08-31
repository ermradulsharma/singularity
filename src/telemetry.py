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
            
        sys.stdout.write(f"[{level.upper()}] [{module}] {message}\n")

logger = CognitiveTelemetry()

