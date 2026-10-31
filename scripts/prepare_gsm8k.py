#!/usr/bin/env python3
"""Build tasks from openai/gsm8k (config main).

Train split -> data/processed/tasks rows with split=train (GRPO).
Test split  -> split=test (before/after only; never train on these).
Also writes data/benchmarks/gsm8k_test.parquet (full test) and gsm8k_test_200.parquet.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "environments" / "gsm8k_v1"))

from gsm8k_v1.rewards import gold_from_solution  # noqa: E402

load_dotenv(ROOT / ".env")


def main() -> None:
    from datasets import load_dataset
    import pandas as pd

    token = os.environ.get("HF_TOKEN")
    ds = load_dataset("openai/gsm8k", "main", token=token)
    rows = []
    for split in ("train", "test"):
        for i, ex in enumerate(ds[split]):
            gold = gold_from_solution(ex["answer"])
            rows.append(
                {
                    "task_id": f"gsm8k:{split}:{i}",
                    "split": split,
                    "prompt": ex["question"],
                    "solution": ex["answer"],
                    "gold": gold,
                    "gold_type": "exact",
                }
            )
    df = pd.DataFrame(rows)
    out = ROOT / "data" / "processed"
    bench = ROOT / "data" / "benchmarks"
    out.mkdir(parents=True, exist_ok=True)
    bench.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out / "tasks.parquet", index=False)
    test = df[df["split"] == "test"]
    test.to_parquet(bench / "gsm8k_test.parquet", index=False)
    test.head(200).to_parquet(bench / "gsm8k_test_200.parquet", index=False)
    print(df.groupby("split").size().to_string())
    print(f"wrote {out / 'tasks.parquet'}")


if __name__ == "__main__":
    main()
