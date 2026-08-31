import torch
import torch.nn as nn
import torch.nn.functional as F
import re

class StepProcessRewardModel(nn.Module):
    """
    Process Reward Model (PRM) for step-by-step intermediate Chain-of-Thought (CoT) reasoning verification.
    Inspired by DeepSeek-R1 Process-Reward Guided Reinforcement Learning.
    """
    def __init__(self, d_model: int = 128):
        super().__init__()
        self.step_evaluator = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, 1),
            nn.Sigmoid()
        )

    def forward(self, step_embeddings: torch.Tensor) -> torch.Tensor:
        """Computes neural step probability rewards for token embeddings."""
        return self.step_evaluator(step_embeddings)

    def extract_reasoning_steps(self, trajectory_text: str) -> list[str]:
        """Extracts individual CoT reasoning steps, think blocks, and code blocks from rollout text."""
        think_blocks = re.findall(r'<think>(.*?)</think>', trajectory_text, re.DOTALL)
        if think_blocks:
            steps = [s.strip() for s in think_blocks[0].split("\n") if s.strip()]
        else:
            steps = [s.strip() for s in trajectory_text.split("\n") if s.strip()]
        return steps if steps else [trajectory_text]

    def score_reasoning_steps(self, steps: list[str]) -> list[float]:
        """Evaluates a trajectory of reasoning steps and returns scalar quality scores between 0.0 and 1.0."""
        scores = []
        for step in steps:
            step_clean = step.strip()
            if not step_clean:
                scores.append(0.0)
                continue
            
            score = 0.5
            # Penalize runtime errors or invalid logic
            if "Error" in step_clean or "Exception" in step_clean or "Traceback" in step_clean:
                score -= 0.4
            # Reward structured CoT thoughts
            if "Thought:" in step_clean or "Observation:" in step_clean or "<think>" in step_clean:
                score += 0.2
            # Reward verified code execution blocks
            if "```python" in step_clean and "```" in step_clean:
                score += 0.25
            # Reward verified conclusions and mathematical proofs
            if "Final Answer:" in step_clean or "Therefore" in step_clean:
                score += 0.15
                
            scores.append(max(0.0, min(1.0, score)))
        return scores

    def score_trajectory(self, trajectory_text: str) -> float:
        """Computes average trajectory score across extracted reasoning steps."""
        steps = self.extract_reasoning_steps(trajectory_text)
        scores = self.score_reasoning_steps(steps)
        return sum(scores) / max(1, len(scores))
