import json
import os

class SessionManager:
    """
    Manages the short-term conversation history for Swarm Agents.
    Stores messages in a rolling JSON file to maintain context between prompts.
    """
    def __init__(self, session_id: str = "default_session", max_history: int = 10):
        self.session_id = session_id
        self.max_history = max_history
        self.history_dir = os.path.join("data", "sessions")
        self.filepath = os.path.join(self.history_dir, f"{self.session_id}.json")
        
        os.makedirs(self.history_dir, exist_ok=True)
        self.history = self._load_history()

    def _load_history(self) -> list:
        if os.path.exists(self.filepath):
            try:
                with open(self.filepath, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                return []
        return []

    def _save_history(self):
        with open(self.filepath, "w", encoding="utf-8") as f:
            json.dump(self.history, f, indent=4)

    def add_message(self, role: str, content: str):
        """Adds a message to the conversation history (e.g., 'user', 'agent', 'observation')."""
        self.history.append({"role": role, "content": content})
        if len(self.history) > self.max_history:
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

    def clear_history(self):
        """Wipes the short-term memory."""
        self.history = []
        self._save_history()
