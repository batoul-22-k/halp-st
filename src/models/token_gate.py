from __future__ import annotations

import torch
from torch import nn


class SemanticTokenGate(nn.Module):
    """Masked token attention over fused token representations."""

    def __init__(self, hidden_size: int, attention_hidden_size: int | None = None):
        super().__init__()
        attention_hidden_size = attention_hidden_size or hidden_size
        self.scorer = nn.Sequential(
            nn.Linear(hidden_size, attention_hidden_size),
            nn.Tanh(),
            nn.Linear(attention_hidden_size, 1, bias=False),
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for module in self.scorer:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight, gain=0.1)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, fused_tokens: torch.Tensor, attention_mask: torch.Tensor):
        mask = attention_mask.bool()
        scores = self.scorer(fused_tokens).squeeze(-1)
        scores = scores.masked_fill(~mask, torch.finfo(scores.dtype).min)
        beta = torch.softmax(scores, dim=-1)
        beta = beta.masked_fill(~mask, 0.0)
        normalizer = beta.sum(dim=-1, keepdim=True).clamp_min(1e-9)
        beta = beta / normalizer
        sentence = (beta.unsqueeze(-1) * fused_tokens).sum(dim=1)
        return sentence, beta

