import sys
import time

from src.telemetry import logger

class RealTimeStreamingVisualizer:
    """Real-Time Terminal Streaming Visualizer for model thoughts, Docker execution panels, and PRM step badges."""
    
    @staticmethod
    def print_section_header(title: str) -> None:
        """Renders section header banner via telemetry logging."""
        msg = f" 🧠 [SINGULARITY REAL-TIME VISUALIZER] :: {title.upper()}"
        logger.log("INFO", "VISUALIZER", f"=== {msg} ===")

    @staticmethod
    def stream_thought_token(token: str) -> None:
        """Streams inner monologue thoughts in real-time token by token with clean character filtering."""
        if not token or token == "\ufffd":
            return
        clean_tok = "".join(c for c in token if c.isprintable() or c in "\n\t ")
        if clean_tok:
            try:
                sys.stdout.write(clean_tok)
                sys.stdout.flush()
            except Exception:
                pass

    @staticmethod
    def render_docker_execution_panel(code_str: str, docker_output: str) -> None:
        """Renders a live visual execution panel showing code sent to Docker and stdout result with responsive text-wrapping."""
        import textwrap
        panel_log = f"Docker Exec Panel | Code: {code_str[:60]}... | Output: {docker_output[:60]}..."
        logger.log("INFO", "VISUALIZER", panel_log)

    @staticmethod
    def render_prm_step_score(step_num: int, step_text: str, score: float) -> None:
        """Renders step quality score badge evaluated by Process Reward Model."""
        status_badge = "🟢 PASSED" if score >= 0.7 else ("🟡 WARN" if score >= 0.4 else "🔴 REJECTED")
        msg = f"[PRM Step {step_num}] Score: {score:.2f} | Badge: {status_badge} | Step: {step_text[:40]}..."
        logger.log("INFO", "PRM_VISUALIZER", msg)
