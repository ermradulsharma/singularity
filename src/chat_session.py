import json
import os

from typing import List, Dict, Any, Optional

class SessionManager:
    """
    Manages the short-term conversation history for Swarm Agents.
    Stores messages in a rolling JSON file to maintain context between prompts.
    """
    def __init__(self, session_id: str = "default_session", max_history: int = 10) -> None:
        self.session_id: str = session_id
        self.max_history: int = max_history
        self.history_dir: str = os.path.join("data", "sessions")
        self.filepath: str = os.path.join(self.history_dir, f"{self.session_id}.json")
        
        os.makedirs(self.history_dir, exist_ok=True)
        self.history: List[Dict[str, Any]] = self._load_history()

    def _load_history(self) -> List[Dict[str, Any]]:
        if os.path.exists(self.filepath):
            try:
                with open(self.filepath, "r", encoding="utf-8") as f:
                    return json.load(f)
            except (json.JSONDecodeError, OSError) as e:
                from src.telemetry import logger
                logger.log("WARNING", "SESSION", f"Failed to load session history from {self.filepath}: {e}")
                return []
        return []

    def _save_history(self) -> None:
        with open(self.filepath, "w", encoding="utf-8") as f:
            json.dump(self.history, f, indent=4)

    def add_message(self, role: str, content: str, block_size: int = 1048576) -> None:
        """Adds a message to history with structural user_input isolation and 90% block_size middle-context pruning."""
        formatted_content = content
        if role.lower() == "user" and not (content.startswith("<user_input>") and content.endswith("</user_input>")):
            formatted_content = f"<user_input>\n{content}\n</user_input>"
            
        self.history.append({"role": role, "content": formatted_content})
        
        # Enforce 90% context budget middle-context pruning
        max_allowed_tokens = int(0.90 * block_size)
        total_tokens = sum(len(m["content"].split()) * 2 for m in self.history)
        
        if total_tokens > max_allowed_tokens and len(self.history) > 3:
            # Preserve system context (index 0) and latest task instruction (index -1), prune middle context
            system_msg = self.history[0]
            latest_msg = self.history[-1]
            middle_msgs = self.history[1:-1]
            pruned_middle = middle_msgs[-(len(middle_msgs) // 2):]
            self.history = [system_msg] + pruned_middle + [latest_msg]
        elif len(self.history) > self.max_history:
            self.history = self.history[-self.max_history:]
            
        self._save_history()

    def get_formatted_history(self) -> str:
        """Returns the history formatted as a string for the LLM context window."""
        if not self.history:
            return "No previous conversation history."
        
        formatted = ""
        for msg in self.history:
            formatted += f"[{msg['role'].upper()}]: {msg['content']}\n"
        return formatted

    def clear_history(self) -> None:
        """Wipes the short-term memory."""
        self.history = []
        self._save_history()
