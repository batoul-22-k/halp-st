import torch

from src.models.pooling import l2_normalize, masked_max_pool, masked_mean_pool


def test_pooling_ignores_padding():
    h = torch.tensor([[[1.0, 2.0], [3.0, 4.0], [100.0, 100.0]]])
    mask = torch.tensor([[1, 1, 0]])
    assert torch.allclose(masked_mean_pool(h, mask), torch.tensor([[2.0, 3.0]]))
    assert torch.allclose(masked_max_pool(h, mask), torch.tensor([[3.0, 4.0]]))
    assert torch.allclose(l2_normalize(torch.tensor([[3.0, 4.0]])).norm(dim=-1), torch.ones(1))

