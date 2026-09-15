from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from src.models import HALPSTModel


def topk(values, k=3):
    vals, idx = torch.topk(values, k=min(k, values.numel()))
    return ", ".join(f"L{int(i)+1}={float(v):.2f}" for v, i in zip(vals, idx))


def safe_token(token: str) -> str:
    return token.encode("ascii", errors="replace").decode("ascii")


def main():
    reference = "A firewall blocks unauthorized network traffic."
    student = "It prevents unapproved traffic from entering the network."
    model = HALPSTModel()
    model.eval()
    batch = model._tokenize([reference, student])
    with torch.no_grad():
        out = model(**batch, return_analysis=True)
    print("input shape:", tuple(batch["input_ids"].shape))
    print("number of hidden layers:", len(out["hidden_states"]) - 1)
    print("hidden state shapes:", [tuple(h.shape) for h in out["hidden_states"][-12:]])
    print("layer attention shape:", tuple(out["layer_attention"].shape))
    print("token attention shape:", tuple(out["token_attention"].shape))
    print("sentence embedding shape:", tuple(out["sentence_embedding"].shape))
    print("embedding norm:", out["sentence_embedding"].norm(dim=-1).tolist())
    sim = torch.sum(out["sentence_embedding"][0] * out["sentence_embedding"][1]).item()
    print("cosine similarity:", sim)
    print("lambda value:", out["lambda"])
    tokens = [safe_token(t) for t in model.tokenizer.convert_ids_to_tokens(batch["input_ids"][0])]
    print("top layer weights per token:")
    for tok, weights, mask in zip(tokens, out["layer_attention"][0], batch["attention_mask"][0]):
        if int(mask):
            print(f"{tok:16s} {topk(weights)}")
    print("top token weights:")
    vals, idx = torch.topk(out["token_attention"][0], k=min(8, int(batch["attention_mask"][0].sum())))
    for value, i in zip(vals, idx):
        print(f"{tokens[int(i)]:16s} {float(value):.3f}")


if __name__ == "__main__":
    main()
