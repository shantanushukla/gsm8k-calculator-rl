#!/usr/bin/env python3
"""Sample a few correct train traces for the wrong-first-formula bucket.

Keeps a sample only when its last formula equals the official number and its
formulas are not the stored solution. The test split is not read.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "environments" / "gsm8k_v1"))
sys.path.insert(0, str(ROOT / "scripts"))

import pyarrow.parquet as pq  # noqa: E402
from mlx_lm import stream_generate  # noqa: E402
from mlx_lm.sample_utils import make_sampler  # noqa: E402

from gsm8k_v1.tools import (  # noqa: E402
    TOOL_SYSTEM,
    calc_spans,
    predicted_formulas,
    reference_expressions,
    tool_match,
    tool_trace,
)
from model_io import chat_prompt, encode, load_policy  # noqa: E402

OUT = ROOT / "results" / "probes"


def to_annotations(text: str) -> str | None:
    body = re.sub(r"\n?<answer>.*", "", text, flags=re.S).strip()
    spans = calc_spans(body)
    if not spans:
        return None
    pieces: list[str] = []
    cursor = 0
    for start, stop, canon in spans:
        if not canon:
            return None
        value = re.match(r"\s*=\s*(-?\d+(?:\.\d+)?)", body[stop:])
        if value is None:
            return None
        raw = body[start + len("calc(") : stop - 1]
        pieces.append(body[cursor:start])
        pieces.append(f"<<{raw}={value.group(1).strip()}>>{value.group(1).strip()}")
        cursor = stop + value.end()
    pieces.append(body[cursor:])
    return "".join(pieces) + "\n"


def main() -> None:
    ids = [line.strip() for line in (OUT / "v12d_source_ids.txt").read_text().splitlines() if line.strip()]
    tasks = {
        row["task_id"]: row
        for row in pq.read_table(ROOT / "data" / "processed" / "tasks.parquet").to_pylist()
    }
    model, tokenizer = load_policy(
        "mlx-community/Llama-3.2-1B-Instruct-4bit",
        str(ROOT / "results" / "adapters" / "gsm8k-sft-v7"),
    )
    sampler = make_sampler(temp=0.8)
    kept = []
    for index, task_id in enumerate(ids, start=1):
        task = tasks[task_id]
        prompt = chat_prompt(tokenizer, str(task["prompt"]), system=TOOL_SYSTEM)
        refs = reference_expressions(task["solution"], str(task["gold"]))
        hits = 0
        for sample_id in range(4):
            pieces = []
            for response in stream_generate(
                model,
                tokenizer,
                encode(tokenizer, prompt),
                max_tokens=256,
                sampler=sampler,
            ):
                pieces.append(response.text)
                if response.finish_reason in ("stop", "length"):
                    break
            text = "".join(pieces)
            if tool_match(text, str(task["gold"])) != 1.0:
                continue
            if predicted_formulas(text) == refs:
                continue
            solution = to_annotations(text)
            if solution is None or tool_trace(solution, str(task["gold"])) is None:
                continue
            hits += 1
            kept.append(
                {
                    "task_id": f"{task_id}:rft{sample_id}",
                    "split": "train",
                    "prompt": task["prompt"],
                    "solution": solution.rstrip() + f"\n#### {task['gold']}",
                    "gold": str(task["gold"]),
                    "loss_span": "all",
                }
            )
            break
        print(f"{index}/{len(ids)} {task_id} kept={hits} total={len(kept)}", flush=True)
    manifest = json.loads((OUT / "v12_manifest.json").read_text())
    for task_id in manifest["replay"]:
        task = tasks[task_id]
        kept.append(
            {
                "task_id": task_id,
                "split": "train",
                "prompt": task["prompt"],
                "solution": task["solution"],
                "gold": str(task["gold"]),
                "loss_span": "all",
            }
        )
    path = OUT / "v12d_rft.jsonl"
    path.write_text("".join(json.dumps(row) + "\n" for row in kept), encoding="utf-8")
    print(f"wrote {path} rows {len(kept)} sampled {len(kept) - len(manifest['replay'])}", flush=True)


if __name__ == "__main__":
    main()
