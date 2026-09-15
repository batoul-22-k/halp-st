from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib.pyplot as plt
import torch

from src.models import HALPSTModel


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sentence", default="A firewall blocks unauthorized network traffic.")
    parser.add_argument("--output-dir", default="results/attention")
    args = parser.parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model = HALPSTModel()
    model.eval()
    batch = model._tokenize([args.sentence])
    with torch.no_grad():
        out = model(**batch, return_analysis=True)
    mask = batch["attention_mask"][0].bool()
    tokens = [t.encode("ascii", errors="replace").decode("ascii") for t in model.tokenizer.convert_ids_to_tokens(batch["input_ids"][0][mask])]
    layer_attn = out["layer_attention"][0][mask].numpy()
    token_attn = out["token_attention"][0][mask].numpy()
    plt.figure(figsize=(10, max(3, len(tokens) * 0.35)))
    plt.imshow(layer_attn, aspect="auto")
    plt.yticks(range(len(tokens)), tokens)
    plt.xticks(range(12), [f"L{i}" for i in range(1, 13)])
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(out_dir / "layer_attention_heatmap.png", dpi=160)
    plt.close()
    plt.figure(figsize=(8, max(3, len(tokens) * 0.35)))
    plt.barh(tokens, token_attn)
    plt.tight_layout()
    plt.savefig(out_dir / "token_attention.png", dpi=160)
    print(f"saved attention plots under {out_dir}")


if __name__ == "__main__":
    main()
