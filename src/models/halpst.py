from __future__ import annotations

import torch
from torch import nn
from transformers import AutoModel, AutoTokenizer

from .baseline import DEFAULT_BACKBONE
from .layer_fusion import TokenAdaptiveLayerFusion
from .pooling import l2_normalize, masked_mean_pool
from .token_gate import SemanticTokenGate


class HALPSTModel(nn.Module):
    def __init__(
        self,
        backbone_name: str = DEFAULT_BACKBONE,
        max_length: int = 128,
        num_transformer_layers: int = 12,
        adaptive_layer_fusion: bool = True,
        semantic_token_gate: bool = True,
        residual_mean: bool = True,
        projection: bool = True,
        residual_mode: str = "lambda",
        freeze_encoder: bool = False,
        encoder: nn.Module | None = None,
        tokenizer=None,
    ):
        super().__init__()
        self.backbone_name = backbone_name
        self.max_length = max_length
        self.num_transformer_layers = num_transformer_layers
        self.adaptive_layer_fusion = adaptive_layer_fusion
        self.semantic_token_gate_enabled = semantic_token_gate
        self.residual_mean = residual_mean
        self.projection_enabled = projection
        self.residual_mode = residual_mode
        self.encoder = encoder or AutoModel.from_pretrained(backbone_name)
        self.tokenizer = tokenizer or AutoTokenizer.from_pretrained(backbone_name)
        self.hidden_size = self.encoder.config.hidden_size

        self.layer_fusion = TokenAdaptiveLayerFusion(self.hidden_size, num_transformer_layers) if adaptive_layer_fusion else None
        self.token_gate = SemanticTokenGate(self.hidden_size) if semantic_token_gate else None
        self.lambda_param = nn.Parameter(torch.tensor(0.1))
        self.gate_param = nn.Parameter(torch.tensor(-2.1972246))
        self.projection = nn.Linear(self.hidden_size, self.hidden_size) if projection else nn.Identity()
        self.layer_norm = nn.LayerNorm(self.hidden_size) if projection else nn.Identity()
        self._init_projection_identity()
        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False

    def _init_projection_identity(self) -> None:
        if isinstance(self.projection, nn.Linear) and self.projection.weight.shape[0] == self.projection.weight.shape[1]:
            nn.init.eye_(self.projection.weight)
            nn.init.zeros_(self.projection.bias)

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

    def _select_transformer_layers(self, hidden_states):
        layers = hidden_states[-self.num_transformer_layers :]
        return torch.stack(layers, dim=2)

    def forward(self, input_ids, attention_mask, token_type_ids=None, return_analysis: bool = False):
        kwargs = {"input_ids": input_ids, "attention_mask": attention_mask}
        if token_type_ids is not None:
            kwargs["token_type_ids"] = token_type_ids
        outputs = self.encoder(**kwargs, output_hidden_states=True, return_dict=True)
        hidden_states = outputs.hidden_states
        last = hidden_states[-1]
        layer_attention = None
        token_attention = None

        if self.adaptive_layer_fusion:
            stacked = self._select_transformer_layers(hidden_states)
            token_repr, layer_attention = self.layer_fusion(stacked, attention_mask)
        else:
            token_repr = last * attention_mask.to(last.dtype).unsqueeze(-1)

        if self.semantic_token_gate_enabled:
            adaptive, token_attention = self.token_gate(token_repr, attention_mask)
        else:
            adaptive = masked_mean_pool(token_repr, attention_mask)

        mean_repr = masked_mean_pool(last, attention_mask)
        if self.residual_mean:
            if self.residual_mode == "lambda":
                fused = mean_repr + self.lambda_param * adaptive
            elif self.residual_mode == "gated":
                gate = torch.sigmoid(self.gate_param)
                fused = (1.0 - gate) * mean_repr + gate * adaptive
            else:
                raise ValueError(f"Unknown residual_mode: {self.residual_mode}")
        else:
            fused = adaptive

        projected = self.layer_norm(self.projection(fused))
        embedding = l2_normalize(projected)
        if return_analysis:
            return {
                "sentence_embedding": embedding,
                "layer_attention": layer_attention,
                "token_attention": token_attention,
                "lambda": self.lambda_param.detach().item(),
                "hidden_states": hidden_states,
            }
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

