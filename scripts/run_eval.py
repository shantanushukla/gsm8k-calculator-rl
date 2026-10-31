#!/usr/bin/env python3
"""Score a model or a completions file on openai/gsm8k test (or train)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "environments" / "gsm8k_v1"))

from gsm8k_v1.rewards import combined, score_task  # noqa: E402

TASKS = ROOT / "data" / "processed" / "tasks.parquet"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--split", default="test")
    p.add_argument("--model", default="")
    p.add_argument("--completions", default="")
    p.add_argument("--out", default="")
    p.add_argument("--n", type=int, default=200)
    args = p.parse_args()

    import pandas as pd

    if not TASKS.exists():
        raise SystemExit("no tasks.parquet — run scripts/prepare_gsm8k.py")
    df = pd.read_parquet(TASKS)
    df = df[df["split"] == args.split].head(args.n)
    if df.empty:
        raise SystemExit(f"no rows for split={args.split}")

    completions: dict[str, str] = {}
    if args.completions:
        with Path(args.completions).open(encoding="utf-8") as f:
            for line in f:
                rec = json.loads(line)
                completions[rec["task_id"]] = rec["completion"]
    elif args.model:
        from openai import OpenAI

        client = OpenAI()
        for _, row in df.iterrows():
            resp = client.chat.completions.create(
                model=args.model,
                messages=[{"role": "user", "content": str(row["prompt"])}],
                max_tokens=512,
            )
            completions[row["task_id"]] = resp.choices[0].message.content or ""
    else:
        raise SystemExit("pass --completions jsonl or --model")

    records = []
    for _, row in df.iterrows():
        comp = completions.get(row["task_id"], "")
        task_s = score_task(comp, str(row["gold"]))
        records.append(
            {
                "task_id": row["task_id"],
                "task_type": "gsm8k",
                "task_score": task_s,
                "reward": combined(comp, str(row["gold"])),
            }
        )
    summary = {
        "dataset": "openai/gsm8k",
        "config": "main",
        "split": args.split,
        "model": args.model or "completions_file",
        "n": len(records),
        "exact_or_task_mean": sum(r["task_score"] for r in records) / len(records),
        "reward_mean": sum(r["reward"] for r in records) / len(records),
        "per_task_type": {"gsm8k": sum(r["task_score"] for r in records) / len(records)},
        "published_reference": "Llama-3.2-1B-Instruct GSM8K 8-shot CoT 44.4% on the full test. This eval is 0-shot.",
        "records": records,
    }
    out = Path(args.out) if args.out else ROOT / "results" / "baselines" / f"gsm8k_{args.split}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({k: summary[k] for k in ("n", "exact_or_task_mean", "reward_mean")}, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
