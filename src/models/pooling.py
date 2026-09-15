import torch


def masked_mean_pool(hidden_states: torch.Tensor, attention_mask: torch.Tensor, eps: float = 1e-9) -> torch.Tensor:
    mask = attention_mask.to(hidden_states.dtype).unsqueeze(-1)
    summed = (hidden_states * mask).sum(dim=1)
    counts = mask.sum(dim=1).clamp_min(eps)
    return summed / counts


def masked_max_pool(hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    mask = attention_mask.bool().unsqueeze(-1)
    masked = hidden_states.masked_fill(~mask, torch.finfo(hidden_states.dtype).min)
    values = masked.max(dim=1).values
    return torch.where(torch.isfinite(values), values, torch.zeros_like(values))


def l2_normalize(embeddings: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    return torch.nn.functional.normalize(embeddings, p=2, dim=-1, eps=eps)

