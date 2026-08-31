import os
import yaml
import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, field_validator
import tiktoken
import sys
import safetensors.torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.model import GPTLanguageModel

app = FastAPI(title="Independent GenAI Model API", version="1.0", description="Production API for custom Foundation Model")

def load_config(config_path="config/config.yaml"):
    with open(config_path, "r") as file:
        return yaml.safe_load(file)

config = load_config()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = None
tokenizer = None

@app.on_event("startup")
async def startup_event():
    global model, tokenizer
    
    tokenizer = tiktoken.get_encoding("gpt2")
    vocab_size = tokenizer.n_vocab + 6
    
    model = GPTLanguageModel(
        vocab_size=vocab_size,
        n_embd=config['model']['n_embd'],
        n_head=config['model']['n_head'],
        n_kv_head=config['model']['n_kv_head'],
        n_layer=config['model']['n_layer'],
        block_size=config['model']['block_size'],
        num_experts=config['model']['num_experts'],
        num_experts_per_tok=config['model']['num_experts_per_tok'],
        dropout=config['model']['dropout']
    )
    
    model_path = config['paths']['model_save']
    if os.path.exists(model_path):
        model.load_state_dict(safetensors.torch.load_file(model_path, device=str(device)))
    else:
        pass
    model.to(device)
    model.eval()

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

@app.post("/v1/generate", response_model=GenerateResponse)
def generate_text(request: GenerateRequest):
    if model is None:
        raise HTTPException(status_code=500, detail="Model is not loaded properly.")
        
    prompt = f"Instruction: {request.instruction}\n"
    if request.input_text:
        prompt += f"Input: {request.input_text}\n"
    prompt += "Output:"
    
    context_tokens = tokenizer.encode(prompt, disallowed_special=())
    if len(context_tokens) > model.block_size:
        raise HTTPException(status_code=413, detail="Encoded prompt exceeds model context limit.")
    context_tensor = torch.tensor([context_tokens], dtype=torch.long, device=device)
    
    with torch.no_grad():
        generated_idx = model.generate(
            context_tensor, 
            max_new_tokens=request.max_tokens,
            temperature=request.temperature,
            top_k=request.top_k,
            top_p=request.top_p,
            agentic_mode=False
        )[0].tolist()
        
    full_output = tokenizer.decode(generated_idx)
    return GenerateResponse(generated_text=full_output)

