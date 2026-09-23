import torch

from src.models.token_gate import SemanticTokenGate


def test_token_gate_masking_and_sums():
    gate = SemanticTokenGate(384)
    z = torch.randn(2, 6, 384)
    mask = torch.tensor([[1, 1, 0, 0, 0, 0], [1, 1, 1, 1, 0, 0]])
    sent, beta = gate(z, mask)
    assert sent.shape == (2, 384)
    assert beta.shape == (2, 6)
    assert torch.allclose(beta.sum(dim=-1), torch.ones(2), atol=1e-5)
    assert torch.all(beta[mask == 0] == 0)

