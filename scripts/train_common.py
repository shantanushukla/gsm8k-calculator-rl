"""LoRA attach, save, and scale toggle shared by warmup and GRPO."""

from __future__ import annotations

import json
from pathlib import Path

import mlx.core as mx
from mlx.utils import tree_flatten

LORA_KEYS = ["self_attn.q_proj", "self_attn.v_proj"]
LORA_SCALE = 16.0


def attach_lora(model, num_layers: int, rank: int) -> None:
    from mlx_lm.tuner.utils import linear_to_lora_layers

    model.freeze()
    linear_to_lora_layers(
        model,
        num_layers,
        {
            "rank": rank,
            "scale": LORA_SCALE,
            "dropout": 0.0,
            "keys": LORA_KEYS,
        },
    )
    mx.eval(model.parameters())


def trainable_count(model) -> int:
    return sum(v.size for _, v in tree_flatten(model.trainable_parameters()))


def lora_b_abs(model) -> float:
    from mlx_lm.tuner.lora import LoRALinear

    total = mx.array(0.0)
    n = 0
    for _, module in model.named_modules():
        if isinstance(module, LoRALinear):
            total = total + mx.sum(mx.abs(module.lora_b))
            n += 1
    if n == 0:
        raise RuntimeError("no LoRA layers on the model")
    mx.eval(total)
    return float(total)


def set_lora_scale(model, scale: float) -> None:
    from mlx_lm.tuner.lora import LoRALinear

    found = 0
    for _, module in model.named_modules():
        if isinstance(module, LoRALinear):
            module.scale = float(scale)
            found += 1
    if found == 0:
        raise RuntimeError("no LoRA layers on the model")


def save_lora(model, path: Path, num_layers: int, rank: int) -> None:
    path.mkdir(parents=True, exist_ok=True)
    weights = dict(tree_flatten(model.trainable_parameters()))
    if not weights:
        raise RuntimeError("no trainable LoRA weights to save")
    mx.save_safetensors(str(path / "adapters.safetensors"), weights)
    (path / "adapter_config.json").write_text(
        json.dumps(
            {
                "fine_tune_type": "lora",
                "num_layers": num_layers,
                "lora_parameters": {
                    "rank": rank,
                    "scale": LORA_SCALE,
                    "dropout": 0.0,
                    "keys": LORA_KEYS,
                },
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def load_lora_weights(model, path: Path) -> None:
    weights = path / "adapters.safetensors"
    if not weights.exists():
        raise FileNotFoundError(weights)
    model.load_weights(str(weights), strict=False)
    mx.eval(model.parameters())
    if lora_b_abs(model) < 1e-8:
        raise RuntimeError(f"LoRA weights in {weights} are still zero")
