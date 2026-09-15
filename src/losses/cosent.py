import torch
from torch import nn


class CoSENTLoss(nn.Module):
    """CoSENT ranking loss for continuous similarity labels in [0, 1]."""

    def __init__(self, scale: float = 20.0):
        super().__init__()
        self.scale = scale

    def forward(self, embeddings_a: torch.Tensor, embeddings_b: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        scores = torch.sum(embeddings_a * embeddings_b, dim=-1) * self.scale
        labels = labels.view(-1)
        score_diffs = scores[:, None] - scores[None, :]
        label_mask = labels[:, None] < labels[None, :]
        if not label_mask.any():
            return scores.sum() * 0.0
        logits = score_diffs[label_mask]
        logits = torch.cat([torch.zeros(1, device=logits.device, dtype=logits.dtype), logits])
        return torch.logsumexp(logits, dim=0)

