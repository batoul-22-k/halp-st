import torch

from src.models import BaselineSTModel, HALPSTModel
from tests.conftest import TinyEncoder, TinyTokenizer


def test_halpst_forward_shapes_attention_norms():
    model = HALPSTModel(encoder=TinyEncoder(), tokenizer=TinyTokenizer(), max_length=8)
    batch = model._tokenize(["same sentence", "different sentence here"])
    out = model(**batch, return_analysis=True)
    assert out["sentence_embedding"].shape == (2, 384)
    assert out["layer_attention"].shape == (2, 8, 12)
    assert out["token_attention"].shape == (2, 8)
    assert torch.isfinite(out["sentence_embedding"]).all()
    assert torch.allclose(out["sentence_embedding"].norm(dim=-1), torch.ones(2), atol=1e-5)
    valid = batch["attention_mask"].bool()
    assert torch.allclose(out["layer_attention"].sum(dim=-1)[valid], torch.ones(valid.sum()), atol=1e-5)
    assert torch.all(out["token_attention"][~valid] == 0)


def test_identical_sentence_and_batch_consistency():
    model = BaselineSTModel(encoder=TinyEncoder(), tokenizer=TinyTokenizer(), max_length=8)
    emb = model.encode(["alpha beta", "alpha beta"], return_tensor=True)
    assert torch.sum(emb[0] * emb[1]) > 0.999
    one = model.encode(["alpha beta"], return_tensor=True)[0]
    batched = model.encode(["gamma", "alpha beta"], return_tensor=True)[1]
    assert torch.allclose(one, batched, atol=1e-6)

