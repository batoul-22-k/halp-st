import torch
from torch import nn


class CosineMSELoss(nn.Module):
    def forward(self, embeddings_a: torch.Tensor, embeddings_b: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        scores = torch.sum(embeddings_a * embeddings_b, dim=-1)
        return torch.nn.functional.mse_loss((scores + 1.0) / 2.0, labels.float())


class MultipleNegativesRankingLoss(nn.Module):
    def __init__(self, scale: float = 20.0):
        super().__init__()
        self.scale = scale
        self.cross_entropy = nn.CrossEntropyLoss()

    def forward(self, embeddings_a: torch.Tensor, embeddings_b: torch.Tensor, labels=None) -> torch.Tensor:
        logits = embeddings_a @ embeddings_b.T * self.scale
        targets = torch.arange(logits.size(0), device=logits.device)
        return self.cross_entropy(logits, targets)

