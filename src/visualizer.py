import sys
import time

from src.telemetry import logger

class RealTimeStreamingVisualizer:
    """Real-Time Terminal Streaming Visualizer for model thoughts, Docker execution panels, and PRM step badges."""
    
    @staticmethod
    def print_section_header(title: str) -> None:
        msg = f" 🧠 [SINGULARITY REAL-TIME VISUALIZER] :: {title.upper()}"
        logger.log("INFO", "VISUALIZER", msg)
        print("\n" + "═" * 65)
        print(msg)
        print("═" * 65)

    @staticmethod
    def stream_thought_token(token: str) -> None:
        """Streams inner monologue thoughts in real-time token by token."""
        sys.stdout.write(token)
        sys.stdout.flush()

    @staticmethod
    def render_docker_execution_panel(code_str: str, docker_output: str) -> None:
        """Renders a live visual execution panel showing code sent to Docker and stdout result with responsive text-wrapping."""
        import textwrap
        logger.log("INFO", "VISUALIZER", f"Docker execution code: {code_str[:80]}... Output: {docker_output[:80]}...")
        print("\n" + "┌" + "─" * 63 + "┐")
        print("│ 🐳 DOCKER SECURE SANDBOX CODE EXECUTION PANEL                │")
        print("├" + "─" * 63 + "┤")
        for line in code_str.strip().split("\n"):
            wrapped = textwrap.wrap(line, width=58) or [""]
            for sub_l in wrapped:
                print(f"│  > {sub_l:<58} │")
        print("├" + "─" * 63 + "┤")
        print("│ ⚙️ EXECUTION RESULT & OBSERVATION (STDOUT):                   │")
        for line in docker_output.strip().split("\n"):
            wrapped = textwrap.wrap(line, width=58) or [""]
            for sub_l in wrapped:
                print(f"│  $ {sub_l:<58} │")
        print("└" + "─" * 63 + "┘\n")

    @staticmethod
    def render_prm_step_score(step_num: int, step_text: str, score: float) -> None:
        """Renders step quality score badge evaluated by Process Reward Model."""
        status_badge = "🟢 PASSED" if score >= 0.7 else ("🟡 WARN" if score >= 0.4 else "🔴 REJECTED")
        msg = f"[PRM Step {step_num}] Score: {score:.2f} | Badge: {status_badge} | Step: {step_text[:40]}..."
        logger.log("INFO", "PRM_VISUALIZER", msg)
        print(msg)
