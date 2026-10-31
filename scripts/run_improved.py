#!/usr/bin/env python3
"""SFT warmup, mixed-question filter, KL-GRPO, then the same 40-row greedy eval.

Each stage is a subprocess so the 16 GB machine drops the model between them.
Stops before GRPO if the filter cannot find mixed train questions.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = ROOT / ".venv" / "bin" / "python"


def run(args: list[str]) -> None:
    print(">>", " ".join(args), flush=True)
    result = subprocess.run([str(PY), *args], cwd=ROOT)
    if result.returncode != 0:
        raise SystemExit(result.returncode)


def main() -> None:
    run(["scripts/sft_warmup.py", "--steps", "48", "--lr", "1e-5", "--max-seq", "384"])
    run(
        [
            "scripts/benchmark.py",
            "--tag",
            "sft",
            "--adapter",
            "results/adapters/gsm8k-sft",
            "--n",
            "40",
            "--max-tokens",
            "256",
        ]
    )
    run(
        [
            "scripts/select_mixed.py",
            "--scan",
            "40",
            "--min-keep",
            "4",
            "--stop-at",
            "12",
            "--max-tokens",
            "192",
        ]
    )
    run(
        [
            "scripts/train_mlx.py",
            "--steps",
            "20",
            "--lr",
            "1e-6",
            "--kl-beta",
            "0.1",
            "--max-tokens",
            "192",
        ]
    )
    run(
        [
            "scripts/benchmark.py",
            "--tag",
            "grpo",
            "--adapter",
            "results/adapters/gsm8k-grpo",
            "--n",
            "40",
            "--max-tokens",
            "256",
        ]
    )
    scores = {}
    failed = ROOT / "results" / "failed_20step" / "trained.json"
    if failed.exists():
        scores["failed_grpo"] = json.loads(failed.read_text())["exact_match"]
    for tag in ("base", "sft", "grpo"):
        path = ROOT / "results" / "baselines" / f"{tag}.json"
        scores[tag] = json.loads(path.read_text())["exact_match"]
    out = ROOT / "results" / "baselines" / "comparison.json"
    out.write_text(json.dumps(scores, indent=2) + "\n", encoding="utf-8")
    run(["scripts/plot_training.py"])
    print("scores", json.dumps(scores), flush=True)


if __name__ == "__main__":
    main()
