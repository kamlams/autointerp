"""What `extract_features.py` needs to turn a TopK SAE checkpoint into activation records:
an SAE loader, a truncated OLMoE forward, and a hook on the layer the SAE was trained on.

Kept inside the autointerp package so feature extraction does not depend on any other project
code. Two facts that are easy to get wrong and were settled by measurement:

* The hookpoint name `layers.L.post_attention_layernorm` does not say which side of the module an
  SAE was trained on. "moein" SAEs use the module's **output** (the normalised MoE/router input),
  "residmid" SAEs its **input** (the post-attention residual before the norm). Feeding an SAE the
  wrong side gives an FVU of 10-170 instead of 0.02-0.5, so check the FVU when in doubt.
* Two checkpoint layouts exist: sparsify's native one (`encoder.weight [d_z, d_x]`,
  `encoder.bias`, `W_dec`, `b_dec`, with `k` in the neighbouring `cfg.json`) and an early export
  (`W_enc [d_x, d_z]`, `W_dec`, `b_dec`, no encoder bias, `k` in `<run>/config.json`).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import torch
from safetensors.torch import load_file

MODEL_NAME = "allenai/OLMoE-1B-7B-0924"


@dataclass
class TopKSAE:
    """Encoder side of a TopK SAE: pre-activation `(x - b_dec) @ W_enc + b_enc`, keep the top k."""

    W_enc: torch.Tensor  # [d_x, d_z]
    W_dec: torch.Tensor  # [d_z, d_x]
    b_dec: torch.Tensor  # [d_x]
    k: int
    path: str = ""
    b_enc: torch.Tensor | None = None  # [d_z]; None for checkpoints without an encoder bias

    @property
    def d_x(self) -> int:
        return self.W_enc.shape[0]

    @property
    def d_z(self) -> int:
        return self.W_enc.shape[1]

    @classmethod
    def load(cls, path: str | Path, device: str = "cuda", k: int | None = None) -> "TopKSAE":
        path = Path(path)
        tensors = load_file(str(path))
        if k is None:
            k = cls._read_k(path)

        def f32(t: torch.Tensor) -> torch.Tensor:
            return t.to(device=device, dtype=torch.float32)

        if "W_enc" in tensors:
            w_enc, b_enc = f32(tensors["W_enc"]), None
        elif "encoder.weight" in tensors:
            w_enc = f32(tensors["encoder.weight"]).t().contiguous()
            b_enc = f32(tensors["encoder.bias"]) if "encoder.bias" in tensors else None
        else:
            raise KeyError(f"unknown SAE layout, keys {sorted(tensors)} in {path}")
        return cls(W_enc=w_enc, W_dec=f32(tensors["W_dec"]), b_dec=f32(tensors["b_dec"]),
                   k=k, path=str(path), b_enc=b_enc)

    @staticmethod
    def _read_k(path: Path) -> int:
        cfg = path.parent / "cfg.json"  # sparsify: <run>/<hookpoint>/{cfg.json, sae.safetensors}
        if cfg.exists():
            return int(json.loads(cfg.read_text())["k"])
        for parent in path.parents:  # early export: <run>/config.json
            cfg = parent / "config.json"
            if cfg.exists():
                return int(json.loads(cfg.read_text())["sae"]["k"])
        raise FileNotFoundError(f"no cfg.json / config.json above {path}; pass k explicitly")

    def encode_topk(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (values [T, k], indices [T, k]) of the k largest relu pre-activations."""
        z = (x - self.b_dec) @ self.W_enc
        if self.b_enc is not None:
            z = z + self.b_enc
        return torch.relu(z).topk(self.k, dim=-1)


def load_truncated_model(layer: int, device: str):
    """Load OLMoE and drop every layer above `layer`; the lm_head is never needed."""
    from transformers import AutoModelForCausalLM

    try:
        model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, dtype=torch.bfloat16)
    except TypeError:  # transformers < 5 spells it torch_dtype
        model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.bfloat16)
    model.model.layers = model.model.layers[: layer + 1]
    model.config.num_hidden_layers = layer + 1
    base = model.model.to(device).eval()
    base.config.num_hidden_layers = layer + 1
    del model
    for param in base.parameters():
        param.requires_grad_(False)
    return base


class ActivationTap:
    """Capture what an SAE at `layers.L.post_attention_layernorm` was trained on.

    side "h": the module's output (normalised MoE/router input; "moein" SAEs).
    side "m": the module's input  (pre-norm post-attention residual; "residmid" SAEs).
    After a forward pass the activation is in `.value`, shape [batch, seq, d_model].
    """

    def __init__(self, layers, layer: int, side: str):
        if side not in ("h", "m"):
            raise ValueError(f"side must be 'h' or 'm', got {side!r}")
        self.value: torch.Tensor | None = None
        self.side = side
        layers[layer].post_attention_layernorm.register_forward_hook(self._hook)

    def _hook(self, module, inputs, output):  # noqa: ARG002
        self.value = inputs[0] if self.side == "m" else output
