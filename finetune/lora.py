"""LoRA from scratch: a frozen linear layer plus a trainable low-rank update.

h = W x + (alpha / r) * B A x,   with B initialised to zero so step 0 equals the base model.
"""

import math

import torch
from torch import nn


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, r: int = 16, alpha: float | None = None):
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad = False
        self.r = r
        self.alpha = alpha if alpha is not None else 2 * r  # rule of thumb from the article: alpha = 2r
        self.scale = self.alpha / r
        d_out, d_in = base.weight.shape
        self.A = nn.Parameter(torch.empty(r, d_in))
        self.B = nn.Parameter(torch.zeros(d_out, r))
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
        self.merged = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.base(x)
        if not self.merged:
            out = out + self.scale * (x @ self.A.T @ self.B.T)
        return out

    def delta_weight(self) -> torch.Tensor:
        return self.scale * (self.B @ self.A)

    @torch.no_grad()
    def merge(self) -> None:
        """W' = W + (alpha / r) B A; after this the adapter costs nothing at inference."""
        if not self.merged:
            self.base.weight += self.delta_weight()
            self.merged = True


def lora_param_count(d_in: int, d_out: int, r: int) -> int:
    return r * d_in + d_out * r


def trainable_params(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters() if p.requires_grad)
