import os
import json

def get_telemetry_summary() -> str:
    """Fetches a real-time HTTP cognitive telemetry dashboard summary of GRPO RL training metrics and rewards."""
    filepath = os.path.join("data", "telemetry.jsonl")
    if not os.path.exists(filepath):
        return "Telemetry log file not found. System is initializing."
        
    try:
        entries = []
        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    try:
                        entries.append(json.loads(line.strip()))
                    except Exception:
                        pass
                    
        if not entries:
            return "No telemetry records logged yet."
            
        recent = entries[-5:]
        summary_lines = [f"Total Logged Events: {len(entries)}"]
        for e in recent:
            module = e.get('module', 'SYS')
            level = e.get('level', 'INFO')
            msg = e.get('message', '')
            summary_lines.append(f"[{level}][{module}]: {msg[:60]}")
            
        return " | ".join(summary_lines)
    except Exception as e:
        return f"Dashboard error: {str(e)}"
