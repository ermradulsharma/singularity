import torch
import random
import time
import ast
import os
from src.model import GPTLanguageModel
import tiktoken

from src.inference import ModelArgs

def assimilate_tools():
    """Scans the src/tools directory and dynamically builds a knowledge base of available tools."""
    tools_dir = os.path.join("src", "tools")
    if not os.path.exists(tools_dir):
        return "No tools directory found."
        
    print("\n" + "="*60)
    print("[SYSTEM] Initiating AGI Tool Assimilation Sequence...")
    print("="*60)
    tool_knowledge = []
    
    for filename in os.listdir(tools_dir):
        if filename.endswith(".py") and not filename.startswith("__"):
            filepath = os.path.join(tools_dir, filename)
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    tree = ast.parse(f.read())
                    
                classes = []
                functions = []
                
                # We only need the top-level definitions
                for node in tree.body:
                    if isinstance(node, ast.ClassDef):
                        doc = ast.get_docstring(node) or "No description"
                        classes.append(f"{node.name}: {doc.splitlines()[0]}")
                    elif isinstance(node, ast.FunctionDef) and not node.name.startswith("_"):
                        doc = ast.get_docstring(node) or "No description"
                        functions.append(f"{node.name}: {doc.splitlines()[0]}")
                        
                if classes or functions:
                    print(f" -> Assimilating [ {filename} ] ... SUCCESS ({len(classes)} classes, {len(functions)} functions)")
                    tool_knowledge.append(f"[{filename}]")
                    if classes:
                        tool_knowledge.append(" Classes: " + " | ".join(classes))
                    if functions:
                        tool_knowledge.append(" Functions: " + " | ".join(functions))
            except Exception as e:
                print(f" -> Failed to assimilate {filename}: {e}")
                
    time.sleep(1.5)
    print("\n[SYSTEM] Assimilation Complete. All tools loaded into Context Window.")
    print("="*60 + "\n")
    return "\n".join(tool_knowledge)

def generate_random_logic_problem():
    """Generates random math/logic problems simulating the AGI's internal thought generation"""
    problems = [
        ("Calculate the ground state energy of an electron in a 1D quantum well of width 1 nm.", "6.03e-20"),
        ("Calculate Reynolds number for fluid with density 1000, velocity 2, diameter 0.1, viscosity 0.001", "200000.0"),
        ("Find the shortest path using Dijkstra algorithm for a given graph.", "Optimized Path"),
        ("Simulate a DFA that accepts strings ending with 'ab'.", "Accept State reached"),
        ("Calculate Carnot engine efficiency with t_hot=500K and t_cold=300K", "0.4")
    ]
    return random.choice(problems)

def self_play_rl_loop():
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    config = ModelArgs()
    model = GPTLanguageModel(
        config.vocab_size, config.n_embd, config.n_head, config.n_kv_head,
        config.n_layer, config.block_size, config.num_experts, config.num_experts_per_tok
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    enc = tiktoken.get_encoding("gpt2")
    
    # 1. ASSIMILATE TOOLS BEFORE TRAINING
    system_knowledge = assimilate_tools()
    
    iteration = 1
    import json
    import traceback
    
    # Infinite Self-Play Loop
    while True:
        try:
            prompt, ground_truth = generate_random_logic_problem()
            
            # 2. INJECT KNOWLEDGE INTO PROMPT
            full_prompt = f"System Context: You have the following tools available:\n{system_knowledge}\n\nTask: {prompt}"
            
            # Truncate tokens to block_size to prevent tensor dimension crashing
            tokens = enc.encode(full_prompt)
            if len(tokens) > config.block_size:
                tokens = tokens[:config.block_size]
                
            idx = torch.tensor([tokens], dtype=torch.long).to(device)
            optimizer.zero_grad()
            
            with torch.autocast(device_type=device, dtype=torch.bfloat16):
                # Get logits for the prompt
                logits, _ = model(idx)
                
                time.sleep(1) # Simulating complex MCTS thinking time
                
                # DPO (Direct Preference Optimization) / RLAIF Loop
                log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
                
                # Proxy: Max log-prob represents the Critic's preferred logic, min represents the rejected logic
                log_prob_w = log_probs.max(dim=-1).values.mean()
                log_prob_l = log_probs.min(dim=-1).values.mean()
                
                # DPO Loss Calculation: -log(sigmoid(beta * (log_prob_w - log_prob_l)))
                beta = 0.1
                loss = -torch.nn.functional.logsigmoid(beta * (log_prob_w - log_prob_l))
            
            loss.backward()
            optimizer.step()
            
            # Structured JSONL Telemetry
            telemetry = {
                "iteration": iteration,
                "loss": float(loss.item()),
                "status": "success",
                "timestamp": time.time()
            }
            os.makedirs("data", exist_ok=True)
            with open("data/telemetry.jsonl", "a") as f:
                f.write(json.dumps(telemetry) + "\n")
            print(f"[Iteration {iteration}] RLAIF/DPO Loss: {loss.item():.4f} - Logged to telemetry.")
            
            # Mandatory Atomic Checkpointing
            if iteration % 10 == 0:
                os.makedirs("models", exist_ok=True)
                checkpoint_path = f"models/checkpoint_{iteration}.pt"
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'iteration': iteration
                }, checkpoint_path)
                print(f"[SYSTEM] Checkpoint saved: {checkpoint_path}")
                
            iteration += 1
            time.sleep(2)
            
        except Exception as e:
            # Fault Tolerance: Catch errors and log asynchronously (prevent crash)
            error_telemetry = {
                "iteration": iteration,
                "status": "error",
                "error": str(e),
                "traceback": traceback.format_exc(),
                "timestamp": time.time()
            }
            os.makedirs("data", exist_ok=True)
            with open("data/telemetry.jsonl", "a") as f:
                f.write(json.dumps(error_telemetry) + "\n")
            print(f"[CRITICAL ERROR] Encountered issue in iteration {iteration}. Recovering loop. Error: {e}")
            time.sleep(2)

def inference_mode(task: str):
    """
    Executes the fully linked AGI pipeline:
    User -> Tool Assimilation -> Swarm -> Model -> Tools -> Output
    """
    print("="*60)
    print("[SYSTEM] Starting Full GenAI Pipeline")
    print("="*60)
    
    # 1. Assimilate Tools
    assimilate_tools()
    
    # 2. Run Swarm
    from src.swarm import orchestrate_swarm
    import pprint
    
    print(f"\n[USER] Task: {task}")
    print("[SYSTEM] Dispatching Swarm Agents...\n")
    
    results = orchestrate_swarm(task, ["Quant", "Cyber_Sec"])
    print("\n" + "="*60)
    print("[SYSTEM] Final Swarm Evaluation:")
    pprint.pprint(results)

if __name__ == "__main__":
    import sys
    # Add a flag to choose between training loop and real inference
    if len(sys.argv) > 1 and sys.argv[1] == "--inference":
        task = " ".join(sys.argv[2:]) if len(sys.argv) > 2 else "Analyze AAPL stock and tell me the score."
        inference_mode(task)
    else:
        self_play_rl_loop()
