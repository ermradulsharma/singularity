import torch
import torch.nn as nn

class SubBrain(nn.Module):
    """Dynamically gated neural sub-brain adapter with input-conditioned soft gating."""
    def __init__(self, n_embd):
        super().__init__()
        self.adapter = nn.Sequential(nn.Linear(n_embd, n_embd//4), nn.GELU(), nn.Linear(n_embd//4, n_embd))
        self.gate = nn.Linear(n_embd, 1)
        nn.init.normal_(self.adapter[2].weight, std=0.01)
        nn.init.zeros_(self.gate.weight)
        nn.init.constant_(self.gate.bias, -2.0)

    def forward(self, x):
        gating = torch.sigmoid(self.gate(x))
        return gating * self.adapter(x)