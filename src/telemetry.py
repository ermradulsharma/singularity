import json
import time
import os
import sys
import queue
import threading

from typing import Dict, Any, Optional

class CognitiveTelemetry:
    """
    Strict AGI logger that writes structured JSONL events asynchronously via a background worker queue.
    """
    def __init__(self, log_dir: str = "data") -> None:
        self.log_dir: str = log_dir
        os.makedirs(self.log_dir, exist_ok=True)
        self.log_file: str = os.path.join(self.log_dir, "telemetry.jsonl")
        self._queue: queue.Queue = queue.Queue()
        self._worker_thread: threading.Thread = threading.Thread(target=self._file_writer_loop, daemon=True)
        self._worker_thread.start()

    def _file_writer_loop(self) -> None:
        """Background worker thread loop for non-blocking file I/O."""
        while True:
            try:
                event = self._queue.get()
                if event is None:
                    break
                with open(self.log_file, "a", encoding="utf-8") as f:
                    f.write(json.dumps(event) + "\n")
                self._queue.task_done()
            except (OSError, IOError, json.JSONDecodeError) as e:
                sys.stderr.write(f"[TELEMETRY ERROR] File write failed: {e}\n")

    def log(self, level: str, module: str, message: str, **kwargs: Any) -> None:
        """Logs a structured JSON event asynchronously without blocking the main training thread."""
        event: Dict[str, Any] = {
            "timestamp": time.time(),
            "level": level.upper(),
            "module": module,
            "message": message,
            **kwargs
        }
        
        # Enqueue for non-blocking file I/O
        try:
            self._queue.put_nowait(event)
        except queue.Full:
            sys.stderr.write("[TELEMETRY WARNING] Log queue full, event dropped.\n")
            
        try:
            sys.stdout.write(f"[{level.upper()}] [{module}] {message}\n")
            sys.stdout.flush()
        except (UnicodeEncodeError, AttributeError):
            clean_msg = message.encode(getattr(sys.stdout, 'encoding', 'utf-8') or 'utf-8', errors='replace').decode(getattr(sys.stdout, 'encoding', 'utf-8') or 'utf-8')
            sys.stdout.write(f"[{level.upper()}] [{module}] {clean_msg}\n")
            sys.stdout.flush()

logger: CognitiveTelemetry = CognitiveTelemetry()

