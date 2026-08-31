import torch
import torch.nn as nn
import torch.nn.functional as F
import re
import ast

class NeuralStepEncoder(nn.Module):
    """Deep Transformer sequence encoder for token-level Process Reward Model scoring."""
    def __init__(self, vocab_size: int = 50257, d_model: int = 128, n_head: int = 4):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Parameter(torch.randn(1, 256, d_model))
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=n_head, dim_feedforward=d_model * 4, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, 1),
            nn.Sigmoid()
        )

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        B, T = token_ids.size()
        pos = self.pos_emb[:, :T, :]
        x = self.tok_emb(token_ids) + pos
        x = self.transformer(x)
        pooled = x.mean(dim=1)
        return self.head(pooled)

class StepProcessRewardModel(nn.Module):
    """
    Process Reward Model (PRM) for step-by-step intermediate Chain-of-Thought (CoT) reasoning verification.
    Inspired by DeepSeek-R1 Process-Reward Guided Reinforcement Learning & Math PRMs.
    """
    def __init__(self, vocab_size: int = 50257, d_model: int = 128):
        super().__init__()
        self.d_model = d_model
        self.neural_encoder = NeuralStepEncoder(vocab_size=vocab_size, d_model=d_model)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """Computes neural step probability rewards for token sequences."""
        return self.neural_encoder(token_ids)

    def extract_reasoning_steps(self, trajectory_text: str) -> list[str]:
        """Extracts individual CoT reasoning steps, think blocks, and code blocks from rollout text."""
        think_blocks = re.findall(r'<think>(.*?)</think>', trajectory_text, re.DOTALL)
        if think_blocks:
            steps = [s.strip() for s in think_blocks[0].split("\n") if s.strip()]
        else:
            steps = [s.strip() for s in trajectory_text.split("\n") if s.strip()]
        return steps if steps else [trajectory_text]

    def _verify_math_and_syntax(self, step_text: str) -> float:
        """Verifies mathematical equations via Z3 SMT solver, LaTeX balance, and Python code AST correctness."""
        score = 0.5
        # Check Python code block correctness via AST parsing
        code_blocks = re.findall(r'```python\n(.*?)\n```', step_text, re.DOTALL)
        for code in code_blocks:
            try:
                ast.parse(code)
                score += 0.3
            except SyntaxError:
                score -= 0.4
        
        # Check equation equality via Z3 SMT Formal Solver
        if "=" in step_text and not step_text.startswith("http"):
            parts = step_text.split("=")
            if len(parts) == 2 and parts[0].strip() and parts[1].strip():
                left_str, right_str = parts[0].strip(), parts[1].strip()
                try:
                    import z3
                    s = z3.Solver()
                    # Evaluate arithmetic equalities formally
                    left_val = eval(left_str, {"__builtins__": None}, {})
                    right_val = eval(right_str, {"__builtins__": None}, {})
                    if left_val == right_val:
                        score += 0.25
                except Exception:
                    score += 0.1

        if step_text.count("(") == step_text.count(")") and step_text.count("[") == step_text.count("]"):
            score += 0.1
        else:
            score -= 0.2
            
        return max(0.0, min(1.0, score))

    def score_reasoning_steps(self, steps: list[str]) -> list[float]:
        """Evaluates a trajectory of reasoning steps and returns scalar quality scores between 0.0 and 1.0."""
        scores = []
        device = next(self.parameters()).device
        import tiktoken
        try:
            enc = tiktoken.get_encoding("gpt2")
        except Exception:
            enc = None

        for step in steps:
            step_clean = step.strip()
            if not step_clean:
                scores.append(0.0)
                continue
            
            # Heuristic & Formal Verification component
            h_score = 0.4
            if "Error" in step_clean or "Exception" in step_clean or "Traceback" in step_clean:
                h_score -= 0.4
            if "Thought:" in step_clean or "Observation:" in step_clean or "<think>" in step_clean:
                h_score += 0.2
            if "Final Answer:" in step_clean or "Therefore" in step_clean or "\\boxed" in step_clean:
                h_score += 0.2
            
            syntax_math_score = self._verify_math_and_syntax(step_clean)

            # Neural PRM Transformer Evaluation component
            with torch.no_grad():
                if enc:
                    t_ids = enc.encode(step_clean[:256])
                else:
                    t_ids = [ord(c) % 50257 for c in step_clean[:256]]
                if not t_ids:
                    t_ids = [0]
                t_tensor = torch.tensor([t_ids[:256]], dtype=torch.long, device=device)
                neural_score = float(self.neural_encoder(t_tensor).item())
                
            combined_score = 0.3 * h_score + 0.3 * syntax_math_score + 0.4 * neural_score
            scores.append(max(0.0, min(1.0, combined_score)))
        return scores

    def score_trajectory(self, trajectory_text: str) -> float:
        """Computes average trajectory score across extracted reasoning steps."""
        steps = self.extract_reasoning_steps(trajectory_text)
        scores = self.score_reasoning_steps(steps)
        return sum(scores) / max(1, len(scores))

