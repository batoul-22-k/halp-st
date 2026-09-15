from __future__ import annotations

from types import SimpleNamespace

import torch
from torch import nn
from transformers import AutoModel, AutoTokenizer

from .pooling import l2_normalize, masked_max_pool, masked_mean_pool
from .token_gate import SemanticTokenGate


DEFAULT_BACKBONE = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


class BaselineSTModel(nn.Module):
    def __init__(
        self,
        backbone_name: str = DEFAULT_BACKBONE,
        pooling: str = "mean",
        max_length: int = 128,
        freeze_encoder: bool = False,
        encoder: nn.Module | None = None,
        tokenizer=None,
    ):
        super().__init__()
        self.backbone_name = backbone_name
        self.max_length = max_length
        self.pooling = pooling
        self.encoder = encoder or AutoModel.from_pretrained(backbone_name)
        self.tokenizer = tokenizer or AutoTokenizer.from_pretrained(backbone_name)
        self.hidden_size = self.encoder.config.hidden_size
        self.token_gate = SemanticTokenGate(self.hidden_size) if pooling == "attention" else None
        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False

    def _tokenize(self, sentences):
        if isinstance(sentences, str):
            sentences = [sentences]
        return self.tokenizer(
            sentences,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )

    def forward(self, input_ids, attention_mask, token_type_ids=None, return_analysis: bool = False):
        kwargs = {"input_ids": input_ids, "attention_mask": attention_mask}
        if token_type_ids is not None:
            kwargs["token_type_ids"] = token_type_ids
        outputs = self.encoder(**kwargs, output_hidden_states=False, return_dict=True)
        last = outputs.last_hidden_state
        token_attention = None
        if self.pooling == "mean":
            embedding = masked_mean_pool(last, attention_mask)
        elif self.pooling == "max":
            embedding = masked_max_pool(last, attention_mask)
        elif self.pooling == "attention":
            embedding, token_attention = self.token_gate(last, attention_mask)
        else:
            raise ValueError(f"Unknown pooling: {self.pooling}")
        embedding = l2_normalize(embedding)
        if return_analysis:
            return {"sentence_embedding": embedding, "token_attention": token_attention}
        return embedding

    @torch.no_grad()
    def encode(self, sentences, batch_size: int = 32, device: str | torch.device | None = None, return_tensor: bool = False):
        was_training = self.training
        self.eval()
        device = torch.device(device or next(self.parameters()).device)
        self.to(device)
        if isinstance(sentences, str):
            sentences = [sentences]
        chunks = []
        for start in range(0, len(sentences), batch_size):
            encoded = self._tokenize(sentences[start : start + batch_size])
            encoded = {k: v.to(device) for k, v in encoded.items()}
            chunks.append(self.forward(**encoded).cpu())
        if was_training:
            self.train()
        result = torch.cat(chunks, dim=0)
        return result if return_tensor else result.numpy()

