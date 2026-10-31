#!/usr/bin/env python3
"""Teach Llama-3.2-1B-Instruct to call calc(), then score that same model.

Before and after both use the calculator harness. A different model is not
part of the comparison. GRPO runs only if the supervised calculator adapter
is at least as good as the untouched model on the same 40 questions.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = ROOT / ".venv" / "bin" / "python"
BASE = ROOT / "results" / "baselines"
MODEL = "mlx-community/Llama-3.2-1B-Instruct-4bit"


def run(args: list[str]) -> None:
    print(">>", " ".join(args), flush=True)
    result = subprocess.run([str(PY), *args], cwd=ROOT)
    if result.returncode != 0:
        raise SystemExit(result.returncode)


def exact(tag: str) -> float | None:
    path = BASE / f"{tag}.json"
    if not path.exists():
        return None
    doc = json.loads(path.read_text())
    if doc.get("tools"):
        return float(doc["tool_match"])
    return float(doc["exact_match"])


def main() -> None:
    run(
        [
            "scripts/benchmark.py",
            "--model",
            MODEL,
            "--tools",
            "--tag",
            "llama_tool_base",
            "--n",
            "40",
            "--max-tokens",
            "256",
        ]
    )
    run(
        [
            "scripts/benchmark.py",
            "--model",
            MODEL,
            "--adapter",
            "results/adapters/gsm8k-sft-v3",
            "--tools",
            "--tag",
            "llama_sft_v3_tools",
            "--n",
            "40",
            "--max-tokens",
            "256",
        ]
    )
    run(
        [
            "scripts/sft_warmup.py",
            "--model",
            MODEL,
            "--init-adapter",
            "results/adapters/gsm8k-sft-v3",
            "--tool-format",
            "--steps",
            "300",
            "--lr",
            "1e-6",
            "--seed",
            "4",
            "--max-seq",
            "384",
            "--out",
            "results/adapters/gsm8k-tool-v2",
            "--metrics",
            "results/metrics_tool_v2.csv",
        ]
    )
    run(
        [
            "scripts/benchmark.py",
            "--model",
            MODEL,
            "--adapter",
            "results/adapters/gsm8k-tool-v2",
            "--tools",
            "--tag",
            "llama_tool_sft",
            "--n",
            "40",
            "--max-tokens",
            "256",
        ]
    )
    base = exact("llama_tool_base")
    sft = exact("llama_tool_sft")
    print(f"tool base {base} tool sft {sft}", flush=True)
    best_tag = "llama_tool_sft"
    best = sft
    sft_doc = {}
    sft_path = BASE / "llama_tool_sft.json"
    if sft_path.exists():
        sft_doc = json.loads(sft_path.read_text())
    use_grpo = (
        sft is not None
        and base is not None
        and sft > 0.10
        and float(sft_doc.get("tool_call_rate") or 0) >= 0.5
    )
    if use_grpo:
        import pandas as pd

        train = pd.read_parquet(ROOT / "data" / "processed" / "tasks.parquet")
        train = train[train["split"] == "train"].sample(48, random_state=1)
        task_file = ROOT / "results" / "tool_train.json"
        task_file.write_text(json.dumps({"kept": train["task_id"].tolist()}), encoding="utf-8")
        run(
            [
                "scripts/train_mlx.py",
                "--model",
                MODEL,
                "--tools",
                "--init-adapter",
                "results/adapters/gsm8k-tool-v2",
                "--task-file",
                "results/tool_train.json",
                "--steps",
                "12",
                "--lr",
                "5e-7",
                "--kl-beta",
                "0.01",
                "--max-tokens",
                "192",
                "--out",
                "results/adapters/gsm8k-tool-v2-grpo",
                "--metrics",
                "results/metrics_tool_grpo.csv",
            ]
        )
        run(
            [
                "scripts/benchmark.py",
                "--model",
                MODEL,
                "--adapter",
                "results/adapters/gsm8k-tool-v2-grpo",
                "--tools",
                "--tag",
                "llama_tool_grpo",
                "--n",
                "40",
                "--max-tokens",
                "256",
            ]
        )
        grpo = exact("llama_tool_grpo")
        print(f"tool grpo {grpo}", flush=True)
        if grpo is not None and grpo + 1e-9 >= (sft or 0):
            best_tag = "llama_tool_grpo"
            best = grpo
    print(f"best tool checkpoint {best_tag} {best}", flush=True)
    if best is not None and base is not None and best > base:
        adapter = {
            "llama_tool_sft": "results/adapters/gsm8k-tool-v2",
            "llama_tool_grpo": "results/adapters/gsm8k-tool-v2-grpo",
        }[best_tag]
        run(
            [
                "scripts/benchmark.py",
                "--model",
                MODEL,
                "--tools",
                "--tag",
                "llama_tool_base_200",
                "--n",
                "200",
                "--max-tokens",
                "256",
            ]
        )
        run(
            [
                "scripts/benchmark.py",
                "--model",
                MODEL,
                "--adapter",
                adapter,
                "--tools",
                "--tag",
                "llama_tool_best_200",
                "--n",
                "200",
                "--max-tokens",
                "256",
            ]
        )
    print("scores", json.dumps({"base": base, "sft": sft, "best": best, "best_tag": best_tag}), flush=True)


if __name__ == "__main__":
    main()
