#!/usr/bin/env python3
"""GRPO from the best prose adapter, stopped if the 40-row score falls.

Starts at results/adapters/gsm8k-sft-v3. The KL reference stays that adapter
even after a chunk improves. Sampling temperature is 0.4: 0.0 makes every
sample identical, and 0.8 was far from the greedy test. Each chunk uses the
next slice of 80 train questions. A worse chunk is not continued.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = ROOT / ".venv" / "bin" / "python"
BASE = ROOT / "results" / "baselines"
MODEL = "mlx-community/Llama-3.2-1B-Instruct-4bit"
REF = "results/adapters/gsm8k-sft-v3"


def run(args: list[str]) -> None:
    print(">>", " ".join(args), flush=True)
    result = subprocess.run([str(PY), *args], cwd=ROOT)
    if result.returncode != 0:
        raise SystemExit(result.returncode)


def score(tag: str) -> float:
    doc = json.loads((BASE / f"{tag}.json").read_text())
    return float(doc["exact_match"])


def main() -> None:
    import pandas as pd

    train = pd.read_parquet(ROOT / "data" / "processed" / "tasks.parquet")
    train = train[train["split"] == "train"].sample(80, random_state=7)
    task_file = ROOT / "results" / "grpo_v5_train.json"
    task_file.write_text(json.dumps({"kept": train["task_id"].tolist()}), encoding="utf-8")

    parent = REF
    best_score = None
    best_tag = "sft_v3"
    if (BASE / "sft_v3.json").exists():
        best_score = score("sft_v3")
        print(f"start from sft_v3 exact {best_score}", flush=True)
    else:
        run(
            [
                "scripts/benchmark.py",
                "--model",
                MODEL,
                "--adapter",
                REF,
                "--tag",
                "sft_v3",
                "--n",
                "40",
                "--max-tokens",
                "256",
            ]
        )
        best_score = score("sft_v3")

    for chunk, offset in enumerate((0, 20, 40, 60)):
        out = f"results/adapters/gsm8k-grpo-v5-{chunk}"
        tag = f"grpo_v5_{chunk}"
        run(
            [
                "scripts/train_mlx.py",
                "--model",
                MODEL,
                "--init-adapter",
                parent,
                "--ref-adapter",
                REF,
                "--task-file",
                "results/grpo_v5_train.json",
                "--offset",
                str(offset),
                "--steps",
                "20",
                "--temperature",
                "0.4",
                "--lr",
                "5e-7",
                "--kl-beta",
                "0.05",
                "--max-tokens",
                "192",
                "--out",
                out,
                "--metrics",
                f"results/metrics_grpo_v5_{chunk}.csv",
            ]
        )
        run(
            [
                "scripts/benchmark.py",
                "--model",
                MODEL,
                "--adapter",
                out,
                "--tag",
                tag,
                "--n",
                "40",
                "--max-tokens",
                "256",
            ]
        )
        got = score(tag)
        metrics = ROOT / f"results/metrics_grpo_v5_{chunk}.csv"
        updated = 0
        if metrics.exists():
            updated = sum(int(line.split(",")[-1]) for line in metrics.read_text().splitlines()[1:] if line.strip())
        print(f"chunk {chunk} exact {got} best {best_score} updates {updated}", flush=True)
        if updated == 0:
            print("no mixed group, stop. temperature produced identical rewards", flush=True)
            break
        if got + 1e-9 < best_score:
            print(f"score fell, stop. keep {best_tag}", flush=True)
            break
        parent = out
        best_score = got
        best_tag = tag
    print("best", best_tag, best_score, flush=True)


if __name__ == "__main__":
    main()
