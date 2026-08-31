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
                    entries.append(json.loads(line))
                    
        if not entries:
            return "No telemetry records logged yet."
            
        recent = entries[-5:]
        summary_lines = [f"Total Logged Iterations: {len(entries)}"]
        for e in recent:
            summary_lines.append(f"Iteration {e.get('iteration')}: GRPO Loss = {e.get('grpo_loss', 0.0):.4f}, Mean Reward = {e.get('mean_reward', 0.0):.4f}")
            
        return " | ".join(summary_lines)
    except Exception as e:
        return f"Dashboard error: {str(e)}"
