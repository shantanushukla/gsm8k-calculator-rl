#!/usr/bin/env python3
"""One-token contrast from gsm8k-sft-v7. Never reads the test split.

For a train question the model misses, find the first formula that leaves the
reference. Keep the model's own prefix and apply one update there: raise the
gold formula's next token and lower the token the model wrote. No solved-row
replay and no loss on the rest of the trace.
"""

from __future__ import annotations

import csv
import json
import math
import random
import sys
import time
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "environments" / "gsm8k_v1"))
sys.path.insert(0, str(ROOT / "scripts"))

from gsm8k_v1.tools import (  # noqa: E402
    TOOL_SYSTEM,
    calc_spans,
    canonicalize,
    reference_expressions,
    tool_trace,
)
from model_io import chat_prompt, encode, load_policy  # noqa: E402
from train_common import attach_lora, load_lora_weights, save_lora, trainable_count  # noqa: E402

SKIP_LOSS = math.log(2)  # gold already more likely than the written token
CAP = 24


def raw_formulas(traced: str) -> list[str]:
    found: list[str] = []
    for start, stop, canon in calc_spans(traced):
        if canon:
            found.append(traced[start + len("calc(") : stop - 1])
    return found


def gold_raw(traced: str, canon: str, used: set[int]) -> str | None:
    for index, expr in enumerate(raw_formulas(traced)):
        if index in used:
            continue
        if canonicalize(expr) == canon:
            used.add(index)
            return expr
    return None


def contrast_pair(tokenizer, prompt: str, completion: str, gold_expr: str, inner: tuple[int, int]):
    """Shared token prefix, the model's next token, and the gold next token."""
    inner_start, inner_end = inner
    model_text = prompt + completion
    abs_start = len(prompt) + inner_start
    abs_end = len(prompt) + inner_end
    gold_text = model_text[:abs_start] + gold_expr + model_text[abs_end:]
    model_ids = encode(tokenizer, model_text)
    gold_ids = encode(tokenizer, gold_text)
    prompt_len = len(encode(tokenizer, prompt))
    if model_ids[:prompt_len] != encode(tokenizer, prompt) or gold_ids[:prompt_len] != model_ids[:prompt_len]:
        return None
    shared = prompt_len
    limit = min(len(model_ids), len(gold_ids))
    while shared < limit and model_ids[shared] == gold_ids[shared]:
        shared += 1
    if shared >= len(model_ids) or shared >= len(gold_ids) or shared >= 512:
        return None
    pre = tokenizer.decode(model_ids[:shared])
    post = tokenizer.decode(model_ids[: shared + 1])
    if len(post) < len(pre):
        return None
    if len(pre) >= abs_end or len(post) <= abs_start:
        return None
    wrong_id = model_ids[shared]
    gold_id = gold_ids[shared]
    if wrong_id == gold_id:
        return None
    return model_ids[:shared], wrong_id, gold_id


def main() -> None:
    import mlx.optimizers as optim

    manifest = json.loads((ROOT / "results" / "probes" / "v12_manifest.json").read_text())
    banned = {
        task_id
        for key in ("number_probe", "both_probe", "next_probe", "op_probe", "control")
        for task_id in manifest[key]
    }
    baseline = json.loads((ROOT / "results" / "baselines" / "v7_train800.json").read_text())
    print("loading mlx-community/Llama-3.2-1B-Instruct-4bit", flush=True)
    model, tokenizer = load_policy("mlx-community/Llama-3.2-1B-Instruct-4bit")
    attach_lora(model, 16, 16)
    load_lora_weights(model, ROOT / "results" / "adapters" / "gsm8k-sft-v7")
    n_train = trainable_count(model)
    print(f"continued from gsm8k-sft-v7 trainable {n_train}", flush=True)
    if n_train != 1_703_936:
        raise SystemExit(f"unexpected LoRA size {n_train}")

    import pyarrow.parquet as pq

    tasks = {
        row["task_id"]: row
        for row in pq.read_table(
            ROOT / "data" / "processed" / "tasks.parquet",
            columns=["task_id", "split", "prompt", "solution", "gold"],
        ).to_pylist()
        if row["split"] == "train"
    }
    first: list[tuple] = []
    later: list[tuple] = []
    skipped = 0
    for record in baseline["records"]:
        task_id = record["task_id"]
        if record["correct"] == 1 or task_id in banned:
            continue
        task = tasks.get(task_id)
        if task is None or task["split"] != "train":
            skipped += 1
            continue
        refs = reference_expressions(task["solution"], str(task["gold"]))
        traced = tool_trace(str(task["solution"]), str(task["gold"]))
        completion = str(record["completion"])
        if not refs or traced is None:
            skipped += 1
            continue
        used: set[int] = set()
        got = 0
        target = None
        for start, stop, canon in calc_spans(completion):
            if canon and got < len(refs) and canon == refs[got]:
                if gold_raw(traced, refs[got], used) is None:
                    target = None
                    break
                got += 1
                continue
            if canon and got < len(refs):
                target = (start, stop, got)
                break
        if target is None:
            skipped += 1
            continue
        start, stop, step = target
        expr = gold_raw(traced, refs[step], used)
        if expr is None:
            skipped += 1
            continue
        prompt = chat_prompt(tokenizer, str(task["prompt"]), system=TOOL_SYSTEM)
        pair = contrast_pair(tokenizer, prompt, completion, expr, (start + len("calc("), stop - 1))
        if pair is None:
            skipped += 1
            continue
        prefix, wrong_id, gold_id = pair
        item = (task_id, prefix, wrong_id, gold_id, tokenizer.decode([wrong_id]), tokenizer.decode([gold_id]), step)
        (first if step == 0 else later).append(item)

    rng = random.Random(44)
    rng.shuffle(first)
    rng.shuffle(later)
    chosen = first[:CAP] + later[:CAP]
    rng.shuffle(chosen)
    print(
        f"contrast candidates first {len(first)} later {len(later)} chosen {len(chosen)} skipped {skipped}",
        flush=True,
    )
    for item in chosen[:5]:
        print(f"example {item[0]} step {item[6]} model {item[4]!r} gold {item[5]!r}", flush=True)
    if len(chosen) < 8:
        raise SystemExit("fewer than 8 contrast rows")

    out = ROOT / "results" / "adapters" / "gsm8k-sft-v14"
    report = {
        "chosen": [item[0] for item in chosen],
        "first_available": len(first),
        "later_available": len(later),
        "examples": [
            {"task_id": item[0], "step": item[6], "model_token": item[4], "gold_token": item[5]}
            for item in chosen
        ],
    }
    (ROOT / "results" / "probes" / "v14_manifest.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    def loss_fn(model, prefix, wrong_id, gold_id):
        logits = model(mx.array(prefix)[None])[0, -1]
        logp = nn.log_softmax(logits.astype(mx.float32), axis=-1)
        margin = logp[gold_id] - logp[wrong_id]
        return mx.logaddexp(mx.array(0.0), -margin)

    grad_fn = nn.value_and_grad(model, loss_fn)
    optimizer = optim.Adam(learning_rate=5e-7)
    metrics_path = ROOT / "results" / "metrics" / "sft_v14.csv"
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    if metrics_path.exists():
        metrics_path.unlink()
    t0 = time.time()
    applied = 0
    with metrics_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["step", "loss", "task_id", "tokens"])
        writer.writeheader()
        for step, (task_id, prefix, wrong_id, gold_id, _wrong, _gold, _at) in enumerate(chosen):
            loss, grads = grad_fn(model, prefix, wrong_id, gold_id)
            value = float(loss)
            if value != value:
                raise SystemExit(f"contrast loss became nan at step {step}")
            kept = value >= SKIP_LOSS
            if kept:
                optimizer.update(model, grads)
                applied += 1
            mx.eval(model.parameters(), loss)
            writer.writerow({"step": step, "loss": round(value, 4), "task_id": task_id, "tokens": len(prefix)})
            handle.flush()
            print(
                f"contrast {step:03d} loss={value:.4f} tokens={len(prefix)}" + ("" if kept else " skip"),
                flush=True,
            )
    if applied == 0:
        print("no contrast step was applied", flush=True)
        raise SystemExit(2)
    save_lora(model, out, 16, 16)
    (out / "sft_task_ids.json").write_text(json.dumps([item[0] for item in chosen], indent=2), encoding="utf-8")
    print(f"saved {out} in {time.time() - t0:.0f}s applied {applied} of {len(chosen)}", flush=True)


if __name__ == "__main__":
    main()
