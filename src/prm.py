import torch
import torch.nn as nn
import torch.nn.functional as F

class StepProcessRewardModel(nn.Module):
    """Process Reward Model (PRM) for step-by-step intermediate Chain-of-Thought reasoning verification."""
    def __init__(self, d_model: int = 128):
        super().__init__()
        self.step_evaluator = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, 1),
            nn.Sigmoid()
        )

    def forward(self, step_embeddings: torch.Tensor) -> torch.Tensor:
        return self.step_evaluator(step_embeddings)

    def score_reasoning_steps(self, steps: list[str]) -> list[float]:
        """Evaluates a trajectory of reasoning steps and returns scalar quality scores between 0.0 and 1.0."""
        scores = []
        for step in steps:
            step_clean = step.strip()
            if not step_clean:
                scores.append(0.0)
                continue
            
            score = 0.5
            if "Error" in step_clean or "Exception" in step_clean:
                score -= 0.4
            if "Thought:" in step_clean or "Observation:" in step_clean:
                score += 0.2
            if "```python" in step_clean and "```" in step_clean:
                score += 0.25
            if "Final Answer:" in step_clean:
                score += 0.15
                
            scores.append(max(0.0, min(1.0, score)))
        return scores
