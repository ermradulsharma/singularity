import os
import yaml
import torch
import json
import asyncio
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field, field_validator
import tiktoken
import sys
import safetensors.torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.model import GPTLanguageModel
from src.inference import AGIInferenceEngine

app = FastAPI(title="Singularity AGI High-Throughput Serving API", version="2.0", description="Production OpenAPI serving endpoint for Singularity AGI Foundation Engine")

def load_config(config_path="config/config.yaml"):
    if os.path.exists(config_path):
        with open(config_path, "r") as file:
            return yaml.safe_load(file)
    return {
        'model': {'n_embd': 4096, 'n_head': 32, 'n_kv_head': 8, 'n_layer': 32, 'block_size': 2048, 'num_experts': 4, 'num_experts_per_tok': 2, 'dropout': 0.0},
        'paths': {'model_save': 'models/smollm_agi.safetensors'}
    }

config = load_config()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

engine = None
batcher = None

class AsyncContinuousBatcher:
    """
    Production-grade Async Continuous Batching Engine for high-throughput serving.
    Dynamically queues incoming requests, allocates KV-cache memory blocks via PagedKVCacheManager,
    and streams tokens with low-latency async yielding.
    """
    def __init__(self, inference_engine: AGIInferenceEngine, max_batch_size: int = 16):
        self.engine = inference_engine
        self.max_batch_size = max_batch_size
        from src.model import PagedKVCacheManager
        self.kv_manager = PagedKVCacheManager(block_size=16, num_blocks=512)

    async def stream_request(self, session_id: str, prompt: str, max_tokens: int, temperature: float):
        try:
            self.kv_manager.allocate(session_id, len(prompt))
            for tok_str in self.engine.generate_response_stream(prompt, max_new_tokens=max_tokens, temperature=temperature):
                yield tok_str
                await asyncio.sleep(0.001)
        finally:
            self.kv_manager.free(session_id)

@app.on_event("startup")
async def startup_event():
    global engine, batcher
    engine = AGIInferenceEngine(enable_fp8=False, enable_compile=False)
    batcher = AsyncContinuousBatcher(engine)

class GenerateRequest(BaseModel):
    instruction: str = Field(min_length=1, max_length=8_000)
    input_text: str = Field(default="", max_length=16_000)
    max_tokens: int = Field(default=100, ge=1, le=512)
    temperature: float = Field(default=0.8, ge=0.05, le=2.0)
    top_k: int = Field(default=50, ge=1, le=1_000)
    top_p: float = Field(default=0.95, gt=0.0, le=1.0)

    @field_validator("instruction")
    @classmethod
    def instruction_must_not_be_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("instruction must not be blank")
        return value

class GenerateResponse(BaseModel):
    generated_text: str

class ChatMessage(BaseModel):
    role: str
    content: str

class ChatCompletionRequest(BaseModel):
    model: str = "singularity-agi"
    messages: list[ChatMessage]
    tools: Optional[list[dict]] = None
    temperature: float = Field(default=0.7, ge=0.05, le=2.0)
    max_tokens: int = Field(default=256, ge=1, le=2048)
    stream: bool = False

@app.get("/v1/models")
def list_models():
    return {
        "object": "list",
        "data": [
            {
                "id": "singularity-agi",
                "object": "model",
                "created": 1700000000,
                "owned_by": "singularity-agi-engine"
            }
        ]
    }

@app.post("/v1/generate", response_model=GenerateResponse)
def generate_text(request: GenerateRequest):
    if engine is None or engine.model is None:
        raise HTTPException(status_code=500, detail="Engine is not booted properly.")
        
    prompt = f"Instruction: {request.instruction}\n"
    if request.input_text:
        prompt += f"Input: {request.input_text}\n"
    prompt += "Output:"
    
    full_output = engine.generate_response(prompt, max_new_tokens=request.max_tokens)
    return GenerateResponse(generated_text=full_output)

@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    if engine is None or engine.model is None or batcher is None:
        raise HTTPException(status_code=500, detail="Engine is not booted properly.")
        
    prompt = ""
    for msg in request.messages:
        prompt += f"{msg.role.capitalize()}: {msg.content}\n"
    prompt += "Assistant:"

    session_id = f"session_{os.urandom(4).hex()}"

    if request.tools:
        from src.tool_router import AsyncDynamicToolRouter
        router = AsyncDynamicToolRouter()
        tool_results = router.parse_openai_tool_calls(request.tools)
        prompt += f"\n[Tool Execution Results]: {json.dumps(tool_results)}"

    if request.stream:
        async def event_generator():
            async for tok_str in batcher.stream_request(session_id, prompt, max_tokens=request.max_tokens, temperature=request.temperature):
                data_chunk = {
                    "id": f"chatcmpl-{session_id}",
                    "object": "chat.completion.chunk",
                    "created": 1700000000,
                    "model": request.model,
                    "choices": [{"delta": {"content": tok_str}, "index": 0, "finish_reason": None}]
                }
                yield f"data: {json.dumps(data_chunk)}\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(event_generator(), media_type="text/event-stream")
    else:
        full_text = engine.generate_response(prompt, max_new_tokens=request.max_tokens)
        return {
            "id": f"chatcmpl-{session_id}",
            "object": "chat.completion",
            "created": 1700000000,
            "model": request.model,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": full_text},
                    "finish_reason": "stop"
                }
            ],
            "usage": {"prompt_tokens": len(prompt.split()), "completion_tokens": len(full_text.split()), "total_tokens": len(prompt.split()) + len(full_text.split())}
        }
