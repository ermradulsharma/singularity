import os
import sys
import pytest
from pydantic import ValidationError

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.api import GenerateRequest, ChatCompletionRequest, ChatMessage

def test_openai_api_schema_integrity():
    """Validates OpenAI-compatible request and response models in src/api.py."""
    req = ChatCompletionRequest(
        messages=[ChatMessage(role="user", content="Explain quantum computing.")]
    )
    assert req.model == "singularity-agi"
    assert req.messages[0].content == "Explain quantum computing."

def test_generate_request_validation():
    """Validates instruction Pydantic field validators."""
    req = GenerateRequest(instruction="Valid instruction")
    assert req.instruction == "Valid instruction"
    
    with pytest.raises(ValidationError):
        GenerateRequest(instruction="   ")
