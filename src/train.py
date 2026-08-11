import os
import json
import torch
import safetensors.torch
import time
import builtins
from torch.utils.data import Dataset, DataLoader
from inference import ModelArgs, AGIInferenceEngine
from model import GPTLanguageModel
from src.telemetry import logger

# 🚨 STRICT COMPLIANCE OVERRIDE
def _telemetry_logger.log("INFO", "SYSTEM", str(*args, **kwargs)):
    message = " ".join(map(str, args)).replace('=', '').strip()
    if message:
        logger.log("INFO", "TRAIN", message)
builtins.print = _telemetry_print

# ---------------------------------------------------------
# AGI EVOLUTION ENGINE (Self-Training Loop)
# ---------------------------------------------------------

class AGIDataset(Dataset):
    def __init__(self, data_path, tokenizer, max_length=512):
        self.examples = []
        
        print(f"[SYSTEM] Reading Knowledge Base: {data_path}...")
        with open(data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        print("[SYSTEM] Tokenizing Knowledge (This may take a moment)...")
        if isinstance(data, dict) and "data" in data:
            data = data["data"]
            
        for item in data[:5000]: # Limit for demonstration speed
            instruction = item.get("instruction", "")
            input_text = item.get("input", "")
            output = item.get("output", "")
            
            prompt = f"Instruction: {instruction}\nInput: {input_text}\nAnswer: {output}"
            tokens = tokenizer.encode(prompt, add_special_tokens=True)
            
            if len(tokens) > max_length:
                tokens = tokens[:max_length]
            elif len(tokens) < max_length:
                tokens = tokens + [tokenizer.eos_token_id] * (max_length - len(tokens))
                
            self.examples.append(torch.tensor(tokens, dtype=torch.long))

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        x = self.examples[idx][:-1]
        y = self.examples[idx][1:]
        return x, y

def train_agi():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[SYSTEM] Initializing AGI Evolution on {device.upper()}...")
    
    try:
        engine = AGIInferenceEngine()
        model = engine.model
        tokenizer = engine.enc
    except Exception as e:
        print(f"[ERROR] Failed to boot AGI architecture: {e}")
        return
        
    model.train()
    
    kb_dir = os.path.join(os.path.dirname(__file__), "..", "data", "knowledge_base")
    kb_file = os.path.join(kb_dir, "phi4_logic_knowledge.json")
    
    if not os.path.exists(kb_file):
        print(f"[ERROR] Knowledge Base is empty! Could not find {kb_file}.")
        print("[TIP] Run 'python src/tools/assimilation_engine.py' first.")
        return
        
    dataset = AGIDataset(kb_file, tokenizer)
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
    
    # Advanced: Automatic Mixed Precision & Gradient Accumulation
    scaler = torch.amp.GradScaler('cuda', enabled=('cuda' in str(device)))
    accumulation_steps = 4
    
    print("============================================================")
    print("🧬 AGI EVOLUTION STARTED (AMP + Grad Accumulation) 🧬")
    print("============================================================")
    
    epochs = 1
    for epoch in range(epochs):
        for step, (x, y) in enumerate(dataloader):
            x, y = x.to(device), y.to(device)
            
            # Autocast for Mixed Precision
            with torch.amp.autocast(device_type='cuda' if 'cuda' in str(device) else 'cpu'):
                logits, loss, _ = model(x, targets=y)
                loss = loss / accumulation_steps
            
            # Backward pass using Scaler
            scaler.scale(loss).backward()
            
            # Optimization Step (Only update weights every 'accumulation_steps')
            if (step + 1) % accumulation_steps == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
            
            if step % 10 == 0:
                print(f"[Epoch {epoch+1}/{epochs}] [Step {step}/{len(dataloader)}] Loss: {loss.item() * accumulation_steps:.4f}")
                
            if step >= 100: # Break early for demo
                break
                
    save_path = "models/smollm_agi_evolved.safetensors"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    safetensors.torch.save_file(model.state_dict(), save_path)
    
    print("============================================================")
    print("[SUCCESS] Evolution Complete! AGI is now smarter.")
    print(f"[SUCCESS] Upgraded Brain saved to: {save_path}")
    print("============================================================")

if __name__ == "__main__":
    train_agi()
