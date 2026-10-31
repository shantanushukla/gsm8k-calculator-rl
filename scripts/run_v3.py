#!/usr/bin/env python3
"""Broader warmup, replay the samples that scored, then GRPO only if that holds.

A new checkpoint is kept as the parent of the next stage only when its
40-row greedy score is not clearly worse than the best checkpoint so far.
The test split is never used for training.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = ROOT / ".venv" / "bin" / "python"
BASE = ROOT / "results" / "baselines"


def run(args: list[str], optional: bool = False) -> bool:
    print(">>", " ".join(args), flush=True)
    result = subprocess.run([str(PY), *args], cwd=ROOT)
    if result.returncode != 0:
        if optional:
            print(f"optional stage failed with {result.returncode}: {args[0]}", flush=True)
            return False
        raise SystemExit(result.returncode)
    return True


def exact(tag: str) -> float | None:
    path = BASE / f"{tag}.json"
    if not path.exists():
        return None
    return float(json.loads(path.read_text())["exact_match"])


def main() -> None:
    run(
        [
            "scripts/sft_warmup.py",
            "--steps",
            "500",
            "--lr",
            "2e-6",
            "--seed",
            "0",
            "--max-seq",
            "384",
            "--out",
            "results/adapters/gsm8k-sft-v3",
            "--metrics",
            "results/metrics_sft_v3.csv",
        ]
    )
    run(
        [
            "scripts/benchmark.py",
            "--tag",
            "sft_v3",
            "--adapter",
            "results/adapters/gsm8k-sft-v3",
            "--n",
            "40",
            "--max-tokens",
            "256",
        ]
    )
    known = {
        "base": exact("base") or 0.20,
        "sft": exact("sft") or 0.175,
        "sft_v3": exact("sft_v3"),
    }
    adapters = {
        "sft": "results/adapters/gsm8k-sft",
        "sft_v3": "results/adapters/gsm8k-sft-v3",
    }
    ranked = [(name, score) for name, score in known.items() if score is not None and name in adapters]
    parent_name, parent_score = max(ranked, key=lambda item: item[1])
    parent = adapters[parent_name]
    print(f"replay parent {parent_name} score {parent_score}", flush=True)

    skip = str(Path(parent) / "sft_task_ids.json")
    run(
        [
            "scripts/reject_sft.py",
            "--adapter",
            parent,
            "--scan",
            "40",
            "--group-size",
            "4",
            "--max-tokens",
            "192",
            "--skip-file",
            skip,
        ]
    )
    n_targets = sum(1 for _ in open(ROOT / "results" / "rft_targets.jsonl", encoding="utf-8"))
    run(
        [
            "scripts/sft_warmup.py",
            "--init-adapter",
            parent,
            "--targets",
            "results/rft_targets.jsonl",
            "--steps",
            str(n_targets),
            "--lr",
            "1e-6",
            "--seed",
            "1",
            "--out",
            "results/adapters/gsm8k-rft",
            "--metrics",
            "results/metrics_rft.csv",
        ]
    )
    run(
        [
            "scripts/benchmark.py",
            "--tag",
            "rft",
            "--adapter",
            "results/adapters/gsm8k-rft",
            "--n",
            "40",
            "--max-tokens",
            "256",
        ]
    )
    rft_score = exact("rft")
    best = max(parent_score, known["base"])
    print(f"rft {rft_score} best_before {best}", flush=True)

    grpo_parent = "results/adapters/gsm8k-rft"
    grpo_floor = rft_score
    if rft_score is None or rft_score + 0.025 < best:
        print("RFT is worse than the best checkpoint. Iterating with an answer tag instead of GRPO.", flush=True)
        tag_from = parent
        run(
            [
                "scripts/sft_warmup.py",
                "--init-adapter",
                tag_from,
                "--answer-tag",
                "--steps",
                "200",
                "--lr",
                "1e-6",
                "--seed",
                "2",
                "--out",
                "results/adapters/gsm8k-tag",
                "--metrics",
                "results/metrics_tag.csv",
            ]
        )
        run(
            [
                "scripts/benchmark.py",
                "--tag",
                "tag",
                "--adapter",
                "results/adapters/gsm8k-tag",
                "--n",
                "40",
                "--max-tokens",
                "256",
            ]
        )
        tag_score = exact("tag")
        print(f"tag {tag_score}", flush=True)
        if tag_score is not None and (rft_score is None or tag_score >= rft_score):
            grpo_parent = "results/adapters/gsm8k-tag"
            grpo_floor = tag_score

    mixed = json.loads((ROOT / "results" / "mixed_train.json").read_text()).get("kept") or []
    if grpo_floor is not None and grpo_floor + 1e-9 >= best - 0.025 and len(mixed) >= 4:
        ok = run(
            [
                "scripts/train_mlx.py",
                "--init-adapter",
                grpo_parent,
                "--task-file",
                "results/mixed_train.json",
                "--steps",
                "12",
                "--lr",
                "5e-7",
                "--kl-beta",
                "0.05",
                "--max-tokens",
                "192",
                "--out",
                "results/adapters/gsm8k-grpo-v3",
                "--metrics",
                "results/metrics_grpo_v3.csv",
            ],
            optional=True,
        )
        if ok:
            run(
            [
                "scripts/benchmark.py",
                "--tag",
                "grpo_v3",
                "--adapter",
                "results/adapters/gsm8k-grpo-v3",
                "--n",
                "40",
                "--max-tokens",
                "256",
            ]
        )
    else:
        print(
            f"skip GRPO floor {grpo_floor} best {best} mixed {len(mixed)}",
            flush=True,
        )

    scores = {}
    failed = ROOT / "results" / "failed_20step" / "trained.json"
    if failed.exists():
        scores["failed_grpo"] = json.loads(failed.read_text())["exact_match"]
    for tag in ("base", "sft", "grpo", "sft_v3", "rft", "tag", "grpo_v3"):
        value = exact(tag)
        if value is not None:
            scores[tag] = value
    (BASE / "comparison.json").write_text(json.dumps(scores, indent=2) + "\n", encoding="utf-8")
    run(["scripts/plot_training.py"])
    print("scores", json.dumps(scores), flush=True)


if __name__ == "__main__":
    main()
