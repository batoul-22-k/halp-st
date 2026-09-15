from __future__ import annotations

import torch
from torch import nn


class TokenAdaptiveLayerFusion(nn.Module):
    """Learns a separate 12-layer distribution for every token."""

    def __init__(self, hidden_size: int, num_layers: int = 12, attention_hidden_size: int | None = None):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
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

    def forward(self, layer_hidden_states: torch.Tensor, attention_mask: torch.Tensor | None = None):
        if layer_hidden_states.dim() != 4:
            raise ValueError("layer_hidden_states must have shape [B, T, L, H]")
        if layer_hidden_states.size(2) != self.num_layers:
            raise ValueError(f"Expected {self.num_layers} layers, found {layer_hidden_states.size(2)}")

        scores = self.scorer(layer_hidden_states).squeeze(-1)
        alpha = torch.softmax(scores, dim=-1)
        fused = (alpha.unsqueeze(-1) * layer_hidden_states).sum(dim=2)
        if attention_mask is not None:
            fused = fused * attention_mask.to(fused.dtype).unsqueeze(-1)
        return fused, alpha

