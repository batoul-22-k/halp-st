from __future__ import annotations

from types import SimpleNamespace

import torch
from torch import nn


class TinyTokenizer:
    def __call__(self, sentences, padding=True, truncation=True, max_length=8, return_tensors="pt"):
        if isinstance(sentences, str):
            sentences = [sentences]
        ids = []
        masks = []
        for s in sentences:
            toks = [1] + [min(99, len(t) + 2) for t in s.split()][: max_length - 2] + [2]
            toks = toks[:max_length]
            mask = [1] * len(toks)
            while len(toks) < max_length:
                toks.append(0)
                mask.append(0)
            ids.append(toks)
            masks.append(mask)
        return {"input_ids": torch.tensor(ids), "attention_mask": torch.tensor(masks)}

    def convert_ids_to_tokens(self, ids):
        return [f"tok{int(i)}" for i in ids]


class TinyEncoder(nn.Module):
    def __init__(self, hidden_size=384, layers=12, vocab_size=128):
        super().__init__()
        self.config = SimpleNamespace(hidden_size=hidden_size)
        self.emb = nn.Embedding(vocab_size, hidden_size)
        self.layers = nn.ModuleList([nn.Linear(hidden_size, hidden_size) for _ in range(layers)])
        for layer in self.layers:
            nn.init.eye_(layer.weight)
            nn.init.zeros_(layer.bias)

    def forward(self, input_ids, attention_mask=None, output_hidden_states=False, return_dict=True):
        x = self.emb(input_ids)
        hidden = [x]
        for i, layer in enumerate(self.layers):
            x = torch.tanh(layer(x) + (i + 1) * 0.01)
            hidden.append(x)
        return SimpleNamespace(last_hidden_state=x, hidden_states=tuple(hidden))

