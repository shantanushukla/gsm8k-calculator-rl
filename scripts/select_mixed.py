#!/usr/bin/env python3
"""Keep train questions the warmed model solves sometimes, not always or never.

Four samples at temperature 0.8. A question is kept when the number correct
is between 1 and 3. Test rows are never read. Exit code 2 if fewer than
--min-keep questions qualify.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "environments" / "gsm8k_v1"))
sys.path.insert(0, str(ROOT / "scripts"))

from gsm8k_v1.rewards import exact_match  # noqa: E402
from model_io import chat_prompt, load_policy  # noqa: E402

OUT = ROOT / "results" / "mixed_train.json"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="mlx-community/Llama-3.2-1B-Instruct-4bit")
    p.add_argument("--adapter", default=str(ROOT / "results" / "adapters" / "gsm8k-sft"))
    p.add_argument("--skip-file", default=str(ROOT / "results" / "adapters" / "gsm8k-sft" / "sft_task_ids.json"))
    p.add_argument("--group-size", type=int, default=4)
    p.add_argument("--max-tokens", type=int, default=192)
    p.add_argument("--scan", type=int, default=40)
    p.add_argument("--min-keep", type=int, default=4)
    p.add_argument("--stop-at", type=int, default=12)
    p.add_argument("--out", default=str(OUT))
    args = p.parse_args()

    import pandas as pd
    from mlx_lm import generate
    from mlx_lm.sample_utils import make_sampler

    skip: set[str] = set()
    skip_path = Path(args.skip_file)
    if skip_path.exists():
        skip = set(json.loads(skip_path.read_text()))

    df = pd.read_parquet(ROOT / "data" / "processed" / "tasks.parquet")
    df = df[df["split"] == "train"].reset_index(drop=True)
    df = df[~df["task_id"].isin(skip)].reset_index(drop=True)
    if df.empty:
        raise SystemExit("no train rows left to probe")

    print(f"loading {args.model} adapter {args.adapter}", flush=True)
    model, tokenizer = load_policy(args.model, args.adapter)
    sampler = make_sampler(temp=0.8)
    kept: list[str] = []
    detail: dict[str, list[float]] = {}
    scanned = 0
    for _, row in df.iterrows():
        if scanned >= args.scan:
            break
        if len(kept) >= args.stop_at and scanned >= args.min_keep:
            break
        scanned += 1
        prompt = chat_prompt(tokenizer, str(row["prompt"]))
        scores: list[float] = []
        for _ in range(args.group_size):
            text = generate(
                model,
                tokenizer,
                prompt=prompt,
                max_tokens=args.max_tokens,
                sampler=sampler,
                verbose=False,
            )
            scores.append(exact_match(text, str(row["gold"])))
        n_ok = int(sum(scores))
        detail[str(row["task_id"])] = scores
        mark = "keep" if 0 < n_ok < args.group_size else "skip"
        print(f"probe {scanned:03d} {row['task_id']} {n_ok}/{args.group_size} {mark}", flush=True)
        if mark == "keep":
            kept.append(str(row["task_id"]))

    payload = {
        "adapter": args.adapter,
        "group_size": args.group_size,
        "max_tokens": args.max_tokens,
        "scanned": scanned,
        "kept": kept,
        "scores": detail,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"kept {len(kept)} of {scanned} -> {out}", flush=True)
    if len(kept) < args.min_keep:
        raise SystemExit(f"only {len(kept)} mixed train questions; need {args.min_keep}")


if __name__ == "__main__":
    main()
