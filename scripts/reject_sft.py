#!/usr/bin/env python3
"""Sample the warmup on train questions and keep the answers that already score.

This is the local version of on-policy distillation, and of the Dream-RSI
idea that past attempts are a replay pool: failures are stored and not
trained on. A question with no correct sample can contribute its original
solution, capped, so the stage is not empty. The test split is never read.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "environments" / "gsm8k_v1"))
sys.path.insert(0, str(ROOT / "scripts"))

from gsm8k_v1.rewards import boxed_match  # noqa: E402
from model_io import chat_prompt, encode, load_policy  # noqa: E402

REPLAY = ROOT / "results" / "replay_pool.jsonl"
TARGETS = ROOT / "results" / "rft_targets.jsonl"
MIXED = ROOT / "results" / "mixed_train.json"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="mlx-community/Llama-3.2-1B-Instruct-4bit")
    p.add_argument("--adapter", required=True)
    p.add_argument("--scan", type=int, default=40)
    p.add_argument("--group-size", type=int, default=4)
    p.add_argument("--max-tokens", type=int, default=192)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--gold-cap", type=int, default=12)
    p.add_argument("--skip-file", default="")
    args = p.parse_args()

    import pandas as pd
    from mlx_lm import generate
    from mlx_lm.sample_utils import make_sampler

    skip: set[str] = set()
    if args.skip_file and Path(args.skip_file).exists():
        skip = set(json.loads(Path(args.skip_file).read_text()))

    df = pd.read_parquet(ROOT / "data" / "processed" / "tasks.parquet")
    df = df[df["split"] == "train"].reset_index(drop=True)
    df = df[~df["task_id"].isin(skip)].sample(frac=1.0, random_state=args.seed)
    df = df.head(args.scan).reset_index(drop=True)
    if df.empty:
        raise SystemExit("no train rows to sample")

    print(f"loading {args.adapter}", flush=True)
    model, tokenizer = load_policy(args.model, args.adapter)
    sampler = make_sampler(temp=0.8)
    REPLAY.parent.mkdir(parents=True, exist_ok=True)
    targets = []
    mixed = []
    n_correct = 0
    n_gold = 0
    with REPLAY.open("w", encoding="utf-8") as replay:
        for i, row in df.iterrows():
            prompt = chat_prompt(tokenizer, str(row["prompt"]))
            scores = []
            correct_text = None
            for _ in range(args.group_size):
                text = generate(
                    model,
                    tokenizer,
                    prompt=prompt,
                    max_tokens=args.max_tokens,
                    sampler=sampler,
                    verbose=False,
                )
                ids = encode(tokenizer, text)
                truncated = len(ids) >= args.max_tokens
                score = 0.0 if truncated else boxed_match(text, str(row["gold"]))
                scores.append(score)
                replay.write(
                    json.dumps(
                        {
                            "task_id": str(row["task_id"]),
                            "gold": str(row["gold"]),
                            "score": score,
                            "truncated": truncated,
                            "completion": text,
                        }
                    )
                    + "\n"
                )
                if score >= 1.0 and correct_text is None:
                    correct_text = text
            n_ok = int(sum(scores))
            n_correct += n_ok
            mark = "keep" if correct_text else "miss"
            print(f"replay {i+1:03d} {row['task_id']} {n_ok}/{args.group_size} {mark}", flush=True)
            if correct_text:
                targets.append(
                    {
                        "task_id": str(row["task_id"]),
                        "prompt": str(row["prompt"]),
                        "solution": correct_text.strip(),
                        "gold": str(row["gold"]),
                        "source": "student",
                    }
                )
            if 0 < n_ok < args.group_size:
                mixed.append(str(row["task_id"]))
            elif correct_text is None and n_gold < args.gold_cap:
                targets.append(
                    {
                        "task_id": str(row["task_id"]),
                        "prompt": str(row["prompt"]),
                        "solution": str(row["solution"]).strip(),
                        "gold": str(row["gold"]),
                        "source": "gold",
                    }
                )
                n_gold += 1

    with TARGETS.open("w", encoding="utf-8") as handle:
        for row in targets:
            handle.write(json.dumps(row) + "\n")
    MIXED.write_text(
        json.dumps({"adapter": args.adapter, "kept": mixed, "scanned": int(args.scan)}, indent=2),
        encoding="utf-8",
    )
    print(
        f"targets {len(targets)} student {len(targets) - n_gold} gold {n_gold} "
        f"correct_samples {n_correct} mixed {len(mixed)}",
        flush=True,
    )
    if not targets:
        raise SystemExit("replay produced no training targets")


if __name__ == "__main__":
    main()
