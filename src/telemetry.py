import json
import time
import os
import sys
import queue
import threading

class CognitiveTelemetry:
    """
    Strict AGI logger that writes structured JSONL events asynchronously via a background worker queue.
    """
    def __init__(self, log_dir="data"):
        self.log_dir = log_dir
        os.makedirs(self.log_dir, exist_ok=True)
        self.log_file = os.path.join(self.log_dir, "telemetry.jsonl")
        self._queue = queue.Queue()
        self._worker_thread = threading.Thread(target=self._file_writer_loop, daemon=True)
        self._worker_thread.start()

    def _file_writer_loop(self):
        """Background worker thread loop for non-blocking file I/O."""
        while True:
            try:
                event = self._queue.get()
                if event is None:
                    break
                with open(self.log_file, "a", encoding="utf-8") as f:
                    f.write(json.dumps(event) + "\n")
                self._queue.task_done()
            except Exception:
                pass

    def log(self, level: str, module: str, message: str, **kwargs):
        """Logs a structured JSON event asynchronously without blocking the main training thread."""
        event = {
            "timestamp": time.time(),
            "level": level.upper(),
            "module": module,
            "message": message,
            **kwargs
        }
        
        # Enqueue for non-blocking file I/O
        try:
            self._queue.put_nowait(event)
        except Exception:
            pass
            
        try:
            sys.stdout.write(f"[{level.upper()}] [{module}] {message}\n")
            sys.stdout.flush()
        except (UnicodeEncodeError, AttributeError):
            clean_msg = message.encode(getattr(sys.stdout, 'encoding', 'utf-8') or 'utf-8', errors='replace').decode(getattr(sys.stdout, 'encoding', 'utf-8') or 'utf-8')
            sys.stdout.write(f"[{level.upper()}] [{module}] {clean_msg}\n")
            sys.stdout.flush()

logger = CognitiveTelemetry()

