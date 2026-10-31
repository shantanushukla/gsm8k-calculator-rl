#!/usr/bin/env python3
"""Supervised LoRA warmup on openai/gsm8k train solutions. Never touches test.

The target is the worked solution, which already ends in #### <number>.
Loss is next-token cross-entropy on the assistant tokens only.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import time
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "environments" / "gsm8k_v1"))
sys.path.insert(0, str(ROOT / "scripts"))

from model_io import SYSTEM, load_policy, supervised_ids  # noqa: E402
from train_common import attach_lora, save_lora, trainable_count  # noqa: E402

METRICS = ROOT / "results" / "metrics_sft.csv"
ADAPTER = ROOT / "results" / "adapters" / "gsm8k-sft"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="mlx-community/Llama-3.2-1B-Instruct-4bit")
    p.add_argument("--steps", type=int, default=500)
    p.add_argument("--lr", type=float, default=2e-6)
    p.add_argument("--max-seq", type=int, default=384)
    p.add_argument("--lora-rank", type=int, default=8)
    p.add_argument("--lora-layers", type=int, default=8)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--init-adapter", default="")
    p.add_argument("--targets", default="")
    p.add_argument("--answer-tag", action="store_true")
    p.add_argument("--tool-format", action="store_true")
    p.add_argument("--checklist", action="store_true")
    p.add_argument(
        "--loss-span",
        default="all",
        choices=("all", "after-first", "first-only", "next-only", "op-lead", "numbers-only", "operator-only"),
    )
    p.add_argument("--min-loss", type=float, default=0.0, help="Skip the update when the loss is already below this.")
    p.add_argument("--holdout", type=int, default=0)
    p.add_argument("--holdout-seed", type=int, default=1)
    p.add_argument("--save-every", type=int, default=0)
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--metrics", default=str(METRICS))
    p.add_argument("--out", default=str(ADAPTER))
    args = p.parse_args()

    import pandas as pd
    import mlx.optimizers as optim
    import random
    from train_common import load_lora_weights

    if args.targets:
        df = pd.read_json(args.targets, lines=True)
    else:
        df = pd.read_parquet(ROOT / "data" / "processed" / "tasks.parquet")
        df = df[df["split"] == "train"]
    df = df.reset_index(drop=True)
    if df.empty:
        raise SystemExit("no train rows")
    if "split" in df.columns and (df["split"] == "test").any():
        raise SystemExit("refusing to train on the test split")
    if args.holdout > 0:
        if "task_id" not in df.columns:
            raise SystemExit("holdout needs task_id")
        ids = sorted(set(df["task_id"].astype(str)))
        rng_hold = random.Random(args.holdout_seed)
        rng_hold.shuffle(ids)
        held = set(ids[: args.holdout])
        df = df[~df["task_id"].astype(str).isin(held)].reset_index(drop=True)
        print(f"held out {len(held)} train rows seed {args.holdout_seed}", flush=True)

    print(f"loading {args.model}", flush=True)
    model, tokenizer = load_policy(args.model)
    attach_lora(model, args.lora_layers, args.lora_rank)
    if args.init_adapter:
        load_lora_weights(model, Path(args.init_adapter))
        print(f"continued from {args.init_adapter}", flush=True)
    n_train = trainable_count(model)
    print(f"trainable params {n_train}", flush=True)
    if args.lora_rank == 8 and args.lora_layers == 8:
        if not 300_000 <= n_train <= 600_000:
            raise SystemExit(f"expected ~426k LoRA params, got {n_train}")
    elif not 300_000 <= n_train <= 4_000_000:
        raise SystemExit(f"LoRA size {n_train} is outside 300k..4M")
    optimizer = optim.Adam(learning_rate=args.lr)

    from gsm8k_v1.tools import TOOL_CHECKLIST, TOOL_SYSTEM, calc_spans, tool_trace

    def span_bounds(solution: str, mode: str) -> tuple[int, int] | None:
        """Character span of the trained solution text. End is exclusive."""
        if mode == "all":
            return 0, len(solution)
        spans = calc_spans(solution)
        if not spans:
            return None
        def value_end_at(span_end: int) -> int:
            value = re.match(r"\s*=\s*-?\d+(?:\.\d+)?", solution[span_end:])
            return span_end + (value.end() if value else 0)

        value_end = value_end_at(spans[0][1])
        if mode == "after-first":
            if value_end >= len(solution):
                return None
            return value_end, len(solution)
        if mode == "first-only":
            return spans[0][0], value_end
        if mode == "next-only":
            if len(spans) < 2:
                return None
            return spans[1][0], value_end_at(spans[1][1])
        if mode == "op-lead":
            marker = solution.find("The operation is ")
            if marker < 0 or marker > spans[0][0]:
                return None
            return marker, value_end
        if mode in ("numbers-only", "operator-only"):
            return spans[0][0], spans[0][1]
        return None

    def focus_spans(solution: str, mode: str, focus: str) -> list[tuple[int, int]] | None:
        """Character spans to train. End is exclusive."""
        bounds = span_bounds(solution, mode)
        if bounds is None:
            return None
        if mode not in ("numbers-only", "operator-only"):
            return [bounds]
        pieces = [piece for piece in focus.split("|") if piece]
        if not pieces:
            return None
        numeric = mode == "numbers-only"

        def bounded_find(start: int) -> int:
            at = start
            while True:
                found_at = solution.find(piece, at, bounds[1])
                if found_at < 0:
                    return -1
                if not numeric:
                    return found_at
                before = solution[found_at - 1] if found_at > 0 else ""
                after_at = found_at + len(piece)
                after = solution[after_at] if after_at < len(solution) else ""
                if before not in "0123456789." and after not in "0123456789.":
                    return found_at
                at = found_at + 1

        found: list[tuple[int, int]] = []
        cursor = bounds[0]
        for piece in pieces:
            at = bounded_find(cursor)
            if at < 0:
                at = bounded_find(bounds[0])
            if at < 0:
                return None
            found.append((at, at + len(piece)))
            cursor = at + len(piece)
        return found

    def token_cut(full_ids: list[int], full_text: str, char_at: int) -> int:
        lo, hi, best = 0, len(full_ids), 0
        while lo <= hi:
            mid = (lo + hi) // 2
            if len(tokenizer.decode(full_ids[:mid])) <= char_at:
                best = mid
                lo = mid + 1
            else:
                hi = mid - 1
        return best

    tool_system = None
    if args.tool_format:
        tool_system = TOOL_CHECKLIST if args.checklist else TOOL_SYSTEM
    rng = random.Random(args.seed)
    pool: list[tuple[str, list[int], int]] = []
    skipped = 0
    order = list(df.index)
    rng.shuffle(order)
    for idx in order:
        row = df.loc[idx]
        if args.tool_format:
            solution = tool_trace(str(row["solution"]), str(row["gold"]))
            if solution is None:
                skipped += 1
                continue
        else:
            solution = str(row["solution"])
            if args.answer_tag:
                solution = solution.rstrip() + f"\n<answer>{row['gold']}</answer>"
        packed = supervised_ids(tokenizer, str(row["prompt"]), solution, system=tool_system)
        if packed is None:
            skipped += 1
            continue
        token_ids, prompt_len = packed
        mode = str(row["loss_span"]) if "loss_span" in row.index and str(row["loss_span"]) else args.loss_span
        focus = str(row["focus"]) if "focus" in row.index and str(row["focus"]) not in ("", "nan") else ""
        regions = focus_spans(solution, mode, focus)
        if regions is None:
            skipped += 1
            continue
        user = (tool_system or SYSTEM) + "\n\n" + str(row["prompt"])
        full = tokenizer.apply_chat_template(
            [
                {"role": "user", "content": user},
                {"role": "assistant", "content": solution.strip()},
            ],
            tokenize=False,
            add_generation_prompt=False,
        )
        located = full.find(solution.strip())
        if located < 0:
            skipped += 1
            continue
        token_spans: list[tuple[int, int]] = []
        if mode == "all":
            token_spans = [(prompt_len, len(token_ids))]
        else:
            for start, end in regions:
                loss_from = token_cut(token_ids, full, located + start)
                loss_to = token_cut(token_ids, full, located + end)
                if loss_to <= loss_from:
                    continue
                token_spans.append((loss_from, loss_to))
        trained = sum(end - start for start, end in token_spans)
        if trained < 1 or (mode not in ("numbers-only", "operator-only") and trained < 2):
            skipped += 1
            continue
        if len(token_ids) > args.max_seq:
            skipped += 1
            continue
        pool.append((str(row["task_id"]), token_ids, token_spans, trained))
        if len(pool) >= args.steps:
            break
    if not pool:
        raise SystemExit("no train solutions fit in the token budget")
    examples = pool * max(1, args.epochs)
    print(
        f"sft examples {len(pool)} epochs {args.epochs} updates {len(examples)} skipped {skipped} lr {args.lr}",
        flush=True,
    )

    def loss_fn(model, token_ids, token_spans):
        logits = model(mx.array(token_ids)[None])[0, :-1]
        targets = mx.array(token_ids[1:])
        logp = nn.log_softmax(logits.astype(mx.float32), axis=-1)
        nll = -logp[mx.arange(targets.size), targets]
        chunks = []
        for loss_from, loss_to in token_spans:
            start = max(0, loss_from - 1)
            end = min(int(nll.shape[0]), max(start + 1, loss_to - 1))
            if end > start:
                chunks.append(nll[start:end])
        if not chunks:
            return nll[0] * 0
        return mx.mean(mx.concatenate(chunks))

    grad_fn = nn.value_and_grad(model, loss_fn)
    out = Path(args.out)
    metrics_path = Path(args.metrics)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    if metrics_path.exists():
        metrics_path.unlink()
    t0 = time.time()
    with metrics_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["step", "loss", "task_id", "tokens"])
        writer.writeheader()
        applied = 0
        for step, (task_id, token_ids, token_spans, trained) in enumerate(examples):
            loss, grads = grad_fn(model, token_ids, token_spans)
            value = float(loss)
            if value != value:
                raise SystemExit(f"sft loss became nan at step {step}")
            kept = value >= args.min_loss
            if kept:
                optimizer.update(model, grads)
                applied += 1
            mx.eval(model.parameters(), loss)
            writer.writerow(
                {
                    "step": step,
                    "loss": round(value, 4),
                    "task_id": task_id,
                    "tokens": len(token_ids),
                }
            )
            handle.flush()
            print(
                f"sft {step:03d} loss={value:.4f} tokens={len(token_ids)} trained={trained}"
                + ("" if kept else " skip"),
                flush=True,
            )
            if args.save_every and (step + 1) % args.save_every == 0:
                save_lora(model, out, args.lora_layers, args.lora_rank)
                print(f"checkpoint {out} at step {step + 1}", flush=True)

    save_lora(model, out, args.lora_layers, args.lora_rank)
    (out / "sft_task_ids.json").write_text(
        json.dumps([task_id for task_id, _, _, _ in examples], indent=2),
        encoding="utf-8",
    )
    print(f"saved {out} in {time.time() - t0:.0f}s applied {applied} of {len(examples)}", flush=True)


if __name__ == "__main__":
    main()
