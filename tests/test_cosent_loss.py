import torch

from src.losses.cosent import CoSENTLoss


def test_cosent_loss_finite_and_grad():
    loss_fn = CoSENTLoss()
    a = torch.nn.functional.normalize(torch.randn(4, 384), dim=-1).requires_grad_()
    b = torch.nn.functional.normalize(torch.randn(4, 384), dim=-1)
    labels = torch.tensor([0.0, 0.3, 0.7, 1.0])
    loss = loss_fn(a, b, labels)
    loss.backward()
    assert torch.isfinite(loss)
    assert a.grad is not None

