import torch
import torch.nn as nn

class SubBrain(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.adapter = nn.Sequential(nn.Linear(n_embd, n_embd//4), nn.GELU(), nn.Linear(n_embd//4, n_embd))
        nn.init.normal_(self.adapter[2].weight, std=0.01)

    def forward(self, x):
        return self.adapter(x) * 0.1