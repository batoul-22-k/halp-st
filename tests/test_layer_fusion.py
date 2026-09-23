import torch

from src.models.layer_fusion import TokenAdaptiveLayerFusion


def test_layer_fusion_shapes_and_sums():
    module = TokenAdaptiveLayerFusion(hidden_size=384, num_layers=12)
    states = torch.randn(2, 5, 12, 384)
    mask = torch.tensor([[1, 1, 1, 0, 0], [1, 1, 1, 1, 0]])
    fused, alpha = module(states, mask)
    assert fused.shape == (2, 5, 384)
    assert alpha.shape == (2, 5, 12)
    assert torch.allclose(alpha.sum(dim=-1), torch.ones(2, 5), atol=1e-5)
    assert torch.all(fused[mask == 0] == 0)

