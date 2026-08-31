import pytest
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.visualizer import RealTimeStreamingVisualizer

def test_visualizer_formatting(capsys):
    RealTimeStreamingVisualizer.print_section_header("Test Header")
    out_header = capsys.readouterr().out
    assert "REAL-TIME VISUALIZER" in out_header

    RealTimeStreamingVisualizer.stream_thought_token("Hello")
    out_tok = capsys.readouterr().out
    assert out_tok == "Hello"

    RealTimeStreamingVisualizer.render_docker_execution_panel("print(42)", "42")
    out_docker = capsys.readouterr().out
    assert "DOCKER SECURE SANDBOX" in out_docker
    assert "42" in out_docker

    RealTimeStreamingVisualizer.render_prm_step_score(1, "Thought test", 0.95)
    out_prm = capsys.readouterr().out
    assert "PASSED" in out_prm
