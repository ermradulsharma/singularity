import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
from typing import List, Dict, Any, Tuple
from src.model import GPTLanguageModel
from src.prm import StepProcessRewardModel
from src.tokenizer import get_unified_tokenizer

class DPOTrainer:
    """Direct Preference Optimization (DPO) Trainer for Alignment sans Reward Model Explicit Training."""
    def __init__(
        self,
        model: GPTLanguageModel,
        ref_model: GPTLanguageModel = None,
        beta: float = 0.1,
        lr: float = 5e-6,
        device: torch.device = None
    ):
        self.device = device or next(model.parameters()).device
        self.model = model.to(self.device)
        self.beta = beta
        
        if ref_model is None:
            self.ref_model = copy.deepcopy(model).eval()
            for p in self.ref_model.parameters():
                p.requires_grad = False
        else:
            self.ref_model = ref_model.eval()
            for p in self.ref_model.parameters():
                p.requires_grad = False
                
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=0.01)

    def _get_batch_log_ps(self, model: GPTLanguageModel, tokens: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Computes sum of log probabilities over sequence tokens specified by mask."""
        logits, _ = model(tokens)
        log_probs = F.log_softmax(logits[:, :-1, :], dim=-1)
        labels = tokens[:, 1:].unsqueeze(-1)
        token_log_probs = log_probs.gather(-1, labels).squeeze(-1)
        return (token_log_probs * mask[:, 1:]).sum(dim=-1)

    def train_step(self, chosen_tokens: torch.Tensor, rejected_tokens: torch.Tensor, chosen_mask: torch.Tensor, rejected_mask: torch.Tensor) -> Dict[str, float]:
        """Executes a single DPO preference optimization step given chosen and rejected trajectories."""
        self.model.train()
        
        policy_chosen_logps = self._get_batch_log_ps(self.model, chosen_tokens, chosen_mask)
        policy_rejected_logps = self._get_batch_log_ps(self.model, rejected_tokens, rejected_mask)
        
        with torch.no_grad():
            ref_chosen_logps = self._get_batch_log_ps(self.ref_model, chosen_tokens, chosen_mask)
            ref_rejected_logps = self._get_batch_log_ps(self.ref_model, rejected_tokens, rejected_mask)
            
        pi_logratios = policy_chosen_logps - policy_rejected_logps
        ref_logratios = ref_chosen_logps - ref_rejected_logps
        
        logits = pi_logratios - ref_logratios
        dpo_loss = -F.logsigmoid(self.beta * logits).mean()
        
        chosen_rewards = self.beta * (policy_chosen_logps - ref_chosen_logps).detach()
        rejected_rewards = self.beta * (policy_rejected_logps - ref_rejected_logps).detach()
        
        self.optimizer.zero_grad()
        dpo_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()
        
        return {
            "dpo_loss": dpo_loss.item(),
            "chosen_reward": chosen_rewards.mean().item(),
            "rejected_reward": rejected_rewards.mean().item(),
            "reward_margin": (chosen_rewards - rejected_rewards).mean().item()
        }

class RLAIFEngine:
    """RLAIF (Reinforcement Learning from AI Feedback) Autonomous Preference Pair Generator."""
    def __init__(self, model: GPTLanguageModel):
        self.model = model
        self.prm = StepProcessRewardModel(vocab_size=model.vocab_size, d_model=model.graph['tok_emb'].weight.size(1)).to(model.graph['tok_emb'].weight.device)
        self.tokenizer = get_unified_tokenizer()

    def generate_preference_pair(self, prompt_tokens: torch.Tensor, max_gen_tokens: int = 64) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Generates candidate completion pairs, scores via AI feedback judge (PRM + AST SMT), and outputs (chosen, rejected) token tensors."""
        self.model.eval()
        with torch.no_grad():
            cand1 = self.model.generate(prompt_tokens, max_new_tokens=max_gen_tokens, temperature=0.9, agentic_mode=False)
            cand2 = self.model.generate(prompt_tokens, max_new_tokens=max_gen_tokens, temperature=0.4, agentic_mode=False)
            
        text1 = self.tokenizer.decode([t for t in cand1[0].tolist() if t < self.tokenizer.n_vocab])
        text2 = self.tokenizer.decode([t for t in cand2[0].tolist() if t < self.tokenizer.n_vocab])
        
        score1 = self.prm.score_trajectory(text1)
        score2 = self.prm.score_trajectory(text2)
        
        if score1 >= score2:
            chosen, rejected = cand1, cand2
        else:
            chosen, rejected = cand2, cand1
            
        chosen_mask = torch.ones_like(chosen, dtype=torch.float32)
        rejected_mask = torch.ones_like(rejected, dtype=torch.float32)
        
        return chosen, rejected, chosen_mask, rejected_mask
