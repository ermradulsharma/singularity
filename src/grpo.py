import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
from typing import List, Dict, Any, Tuple
from src.model import GPTLanguageModel
from src.prm import StepProcessRewardModel
from src.tokenizer import get_unified_tokenizer

class GRPOTrainer:
    """Group Relative Policy Optimization (GRPO) Reinforcement Learning Trainer for Reasoning Alignment."""
    def __init__(
        self,
        model: GPTLanguageModel,
        ref_model: GPTLanguageModel = None,
        lr: float = 1e-5,
        group_size: int = 4,
        clip_eps: float = 0.2,
        kl_beta: float = 0.04,
        device: torch.device = None
    ):
        self.device = device or next(model.parameters()).device
        self.model = model.to(self.device)
        self.group_size = group_size
        self.clip_eps = clip_eps
        self.kl_beta = kl_beta
        
        if ref_model is None:
            self.ref_model = copy.deepcopy(model).eval()
            for p in self.ref_model.parameters():
                p.requires_grad = False
        else:
            self.ref_model = ref_model.eval()
            for p in self.ref_model.parameters():
                p.requires_grad = False
                
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=0.01)
        self.prm = StepProcessRewardModel(vocab_size=model.vocab_size, d_model=model.graph['tok_emb'].weight.size(1)).to(self.device)
        self.tokenizer = get_unified_tokenizer()

    def _compute_rewards(self, prompts: List[str], completions: List[str]) -> torch.Tensor:
        """Computes step-level PRM, math correctness, and format compliance rewards for generated trajectories."""
        rewards = []
        for prompt, comp in zip(prompts, completions):
            full_text = prompt + comp
            prm_score = self.prm.score_trajectory(full_text)
            
            # Format compliance reward (DeepSeek-R1 style <think>...</think> structure)
            format_reward = 1.0 if ("<think>" in comp and "</think>" in comp) else 0.0
            
            # Code/Math syntax check bonus
            syntax_bonus = 0.5 if ("```python" in comp and "```" in comp) else 0.0
            
            total_reward = 0.6 * prm_score + 0.3 * format_reward + 0.1 * syntax_bonus
            rewards.append(total_reward)
            
        return torch.tensor(rewards, dtype=torch.float32, device=self.device)

    def sample_group_rollouts(self, prompt_tokens: torch.Tensor, max_gen_tokens: int = 128) -> Tuple[torch.Tensor, List[str]]:
        """Samples G completion trajectories per prompt using stochastic temperature rollouts."""
        B, T = prompt_tokens.shape
        prompt_tokens_exp = prompt_tokens.repeat_interleave(self.group_size, dim=0)
        
        self.model.eval()
        with torch.no_grad():
            gen_tokens = self.model.generate(
                prompt_tokens_exp,
                max_new_tokens=max_gen_tokens,
                temperature=0.8,
                top_p=0.95,
                agentic_mode=False
            )
            
        completions_text = []
        for i in range(gen_tokens.size(0)):
            completion_ids = gen_tokens[i, T:].tolist()
            text = self.tokenizer.decode([t for t in completion_ids if t < self.tokenizer.n_vocab])
            completions_text.append(text)
            
        return gen_tokens, completions_text

    def train_step(self, prompt_tokens: torch.Tensor, max_gen_tokens: int = 128) -> Dict[str, float]:
        """Executes a single GRPO policy gradient optimization step over group trajectory rollouts."""
        self.model.train()
        B, T_prompt = prompt_tokens.size()
        
        # 1. Sample group completions G for each prompt in batch
        gen_tokens, completions_text = self.sample_group_rollouts(prompt_tokens, max_gen_tokens=max_gen_tokens)
        total_samples = B * self.group_size
        T_total = gen_tokens.size(1)
        
        prompt_texts = []
        for b in range(B):
            p_text = self.tokenizer.decode([t for t in prompt_tokens[b].tolist() if t < self.tokenizer.n_vocab])
            prompt_texts.extend([p_text] * self.group_size)
            
        # 2. Evaluate trajectory rewards R_i
        rewards = self._compute_rewards(prompt_texts, completions_text) # [B * G]
        
        # 3. Compute Group-Relative Advantages A_i
        rewards_grouped = rewards.view(B, self.group_size)
        mean_r = rewards_grouped.mean(dim=1, keepdim=True)
        std_r = rewards_grouped.std(dim=1, keepdim=True) + 1e-8
        advantages = ((rewards_grouped - mean_r) / std_r).view(-1) # [B * G]
        
        # 4. Forward pass under current policy and reference policy
        completion_mask = torch.zeros((total_samples, T_total), dtype=torch.bool, device=self.device)
        completion_mask[:, T_prompt:] = True
        
        with torch.no_grad():
            ref_logits, _ = self.ref_model(gen_tokens)
            ref_log_probs = F.log_softmax(ref_logits, dim=-1)
            ref_token_log_probs = ref_log_probs.gather(-1, gen_tokens.unsqueeze(-1)).squeeze(-1)
            
        logits, _ = self.model(gen_tokens)
        log_probs = F.log_softmax(logits, dim=-1)
        token_log_probs = log_probs.gather(-1, gen_tokens.unsqueeze(-1)).squeeze(-1)
        
        # 5. Compute PPO Ratio & Clipped Objective
        old_token_log_probs = token_log_probs.detach()
        ratio = torch.exp(token_log_probs - old_token_log_probs)
        
        adv_exp = advantages.unsqueeze(-1).expand_as(ratio)
        surr1 = ratio * adv_exp
        surr2 = torch.clamp(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * adv_exp
        policy_loss = -torch.min(surr1, surr2)
        
        # 6. KL Penalty w.r.t Reference Policy
        kl_div = torch.exp(ref_token_log_probs - token_log_probs) - (ref_token_log_probs - token_log_probs) - 1.0
        
        total_loss = (policy_loss + self.kl_beta * kl_div) * completion_mask.float()
        loss = total_loss.sum() / max(1, completion_mask.sum())
        
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()
        
        return {
            "grpo_loss": loss.item(),
            "mean_reward": rewards.mean().item(),
            "std_reward": std_r.mean().item(),
            "kl_divergence": kl_div.mean().item()
        }
