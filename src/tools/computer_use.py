"""Autonomous OS Computer Use and Desktop Control Tool Module for Singularity AGI Engine."""
import os
import time
import base64
from io import BytesIO
from typing import Dict, Any, Tuple

class OSComputerUseTool:
    """Agentic OS GUI automation engine for screenshot capture, mouse navigation, and keyboard execution."""

    def capture_screen(self, format: str = "PNG") -> Dict[str, Any]:
        """Captures the current OS desktop screenshot and returns image metadata with base64 encoding."""
        try:
            from PIL import ImageGrab
            screenshot = ImageGrab.grab()
            buffer = BytesIO()
            screenshot.save(buffer, format=format)
            img_b64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
            return {
                "status": "success",
                "width": screenshot.width,
                "height": screenshot.height,
                "base64_image": img_b64
            }
        except Exception as e:
            return {"status": "error", "message": f"Screen capture failed: {e}"}

    def execute_mouse_click(self, x: int, y: int, button: str = "left", double: bool = False) -> Dict[str, Any]:
        """Executes mouse cursor movement to target coordinates (x, y) and performs click action."""
        try:
            import pyautogui
            pyautogui.moveTo(x, y)
            if double:
                pyautogui.doubleClick(button=button)
            else:
                pyautogui.click(button=button)
            return {"status": "success", "action": "mouse_click", "x": x, "y": y, "button": button}
        except Exception:
            return {"status": "simulated", "action": "mouse_click", "x": x, "y": y, "button": button}

    def execute_keyboard_type(self, text: str, interval: float = 0.05) -> Dict[str, Any]:
        """Types the specified text string into active desktop window with keypress interval timing."""
        try:
            import pyautogui
            pyautogui.write(text, interval=interval)
            return {"status": "success", "action": "keyboard_type", "chars_typed": len(text)}
        except Exception:
            return {"status": "simulated", "action": "keyboard_type", "chars_typed": len(text)}

    def execute_key_combination(self, keys: list) -> Dict[str, Any]:
        """Executes OS hotkey shortcuts by pressing multiple keys sequentially."""
        try:
            import pyautogui
            pyautogui.hotkey(*keys)
            return {"status": "success", "action": "hotkey", "keys": keys}
        except Exception:
            return {"status": "simulated", "action": "hotkey", "keys": keys}
