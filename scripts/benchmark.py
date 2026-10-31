#!/usr/bin/env python3
"""Score the current MLX weights on openai/gsm8k test. Does not train."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "environments" / "gsm8k_v1"))
sys.path.insert(0, str(ROOT / "scripts"))

from gsm8k_v1.rewards import (  # noqa: E402
    ANSWER_RE,
    BOX_RE,
    HASH_RE,
    boxed_match,
    exact_match,
    format_box,
    normalize_number,
)
from gsm8k_v1.tools import (  # noqa: E402
    TOOL_CHECKLIST,
    TOOL_SYSTEM,
    calculator_insertion,
    cut_repeated_tail,
    expression_f1,
    last_formula_value,
    tool_calls,
    tool_match,
)
from model_io import chat_prompt, encode, few_shot_prompt, load_policy  # noqa: E402


def greedy_with_calculator(model, tokenizer, prompt: str, max_tokens: int) -> str:
    """Greedy decode. When a calc() closes, append the computed value and do not sample it."""
    from mlx_lm import stream_generate

    prompt_ids = encode(tokenizer, prompt)
    completion: list[int] = []
    text = ""
    remaining = max_tokens
    ignored_calc = -1
    while remaining > 0:
        stop = False
        truncated = False
        saw = False
        insert = None
        for response in stream_generate(
            model,
            tokenizer,
            prompt_ids + completion,
            max_tokens=remaining,
        ):
            text += response.text
            if response.finish_reason == "stop":
                stop = True
                break
            completion.append(int(response.token))
            saw = True
            remaining -= 1
            if response.finish_reason == "length" or remaining <= 0:
                truncated = True
                break
            extra = calculator_insertion(text)
            if not extra:
                continue
            calc_at = text.rfind("calc(")
            if calc_at == ignored_calc:
                continue
            full = prompt_ids + completion
            decoded = tokenizer.decode(full, skip_special_tokens=False)
            new_ids = encode(tokenizer, decoded + extra)
            if new_ids[: len(full)] != full:
                ignored_calc = calc_at
                continue
            insert = (new_ids[len(full) :], extra)
            break
        if truncated or stop or not saw or insert is None:
            break
        forced, extra = insert
        completion.extend(forced)
        text += extra
        if len(completion) >= max_tokens:
            break
    return text


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="mlx-community/Llama-3.2-1B-Instruct-4bit")
    p.add_argument("--adapter", default="")
    p.add_argument("--n", type=int, default=50)
    p.add_argument("--split", default="test", choices=("test", "train"))
    p.add_argument("--offset", type=int, default=0)
    p.add_argument("--ids", default="", help="Train or test task ids, one per line. Replaces offset and n.")
    p.add_argument("--max-tokens", type=int, default=256)
    p.add_argument("--tag", default="base")
    p.add_argument("--tools", action="store_true")
    p.add_argument("--execute", action="store_true")
    p.add_argument("--samples", type=int, default=1)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--shots", type=int, default=0)
    p.add_argument("--retrieve", type=int, default=0)
    p.add_argument("--checklist", action="store_true")
    args = p.parse_args()

    import pandas as pd
    from collections import Counter

    from mlx_lm import generate, stream_generate
    from mlx_lm.sample_utils import make_sampler

    def one_completion(prompt: str, temperature: float) -> str:
        if temperature <= 0:
            return generate(model, tokenizer, prompt=prompt, max_tokens=args.max_tokens, verbose=False)
        pieces = []
        for response in stream_generate(
            model,
            tokenizer,
            encode(tokenizer, prompt),
            max_tokens=args.max_tokens,
            sampler=make_sampler(temp=temperature),
        ):
            pieces.append(response.text)
            if response.finish_reason in ("stop", "length"):
                break
        return "".join(pieces)

    def answer_key(text: str) -> str:
        if args.tools:
            return last_formula_value(text)
        matched = ANSWER_RE.search(text or "")
        boxes = BOX_RE.findall(text or "")
        hashes = HASH_RE.findall(text or "")
        raw = matched.group(1) if matched else boxes[-1] if boxes else hashes[-1] if hashes else ""
        return normalize_number(raw) if raw else ""

    df = pd.read_parquet(ROOT / "data" / "processed" / "tasks.parquet")
    exemplars: list[tuple[str, str]] = []
    if args.shots > 0:
        from gsm8k_v1.tools import tool_trace

        train = df[df["split"] == "train"].sample(frac=1, random_state=0)
        for _, sample in train.iterrows():
            answer = str(sample["solution"])
            if args.tools:
                traced = tool_trace(answer, str(sample["gold"]))
                if traced is None:
                    continue
                answer = traced
            exemplars.append((str(sample["prompt"]), answer))
            if len(exemplars) >= args.shots:
                break
        if len(exemplars) < args.shots:
            raise SystemExit(f"only found {len(exemplars)} train exemplars")
    retrieved: list[tuple[str, str]] = []
    if args.retrieve > 0:
        from gsm8k_v1.tools import tool_trace

        def tokens(text: str) -> set[str]:
            return set(re.findall(r"[a-z0-9]+", text.lower()))

        train_rows = df[df["split"] == "train"]
        retrieved = []
        for _, sample in train_rows.iterrows():
            answer = str(sample["solution"])
            if args.tools:
                traced = tool_trace(answer, str(sample["gold"]))
                if traced is None:
                    continue
                answer = traced
            retrieved.append((str(sample["prompt"]), answer, tokens(str(sample["prompt"]))))
        if len(retrieved) < args.retrieve:
            raise SystemExit(f"only found {len(retrieved)} train traces to retrieve")
    df = df[df["split"] == args.split]
    if args.ids:
        wanted = [line.strip() for line in Path(args.ids).read_text(encoding="utf-8").splitlines() if line.strip()]
        order = {task: i for i, task in enumerate(wanted)}
        df = df[df["task_id"].astype(str).isin(order)].copy()
        df["_ord"] = df["task_id"].astype(str).map(order)
        df = df.sort_values("_ord").drop(columns="_ord")
    else:
        df = df.iloc[args.offset : args.offset + args.n]
    df = df.reset_index(drop=True)
    if df.empty:
        raise SystemExit(f"no {args.split} rows at offset {args.offset}")
    model, tokenizer = load_policy(args.model, args.adapter or None)

    hits = 0
    fmt = 0
    boxed_hits = 0
    tool_hits = 0
    expr_sum = 0.0
    calls = 0
    records = []
    system = (TOOL_CHECKLIST if args.checklist else TOOL_SYSTEM) if args.tools else None
    for i, row in df.iterrows():
        chosen_examples = exemplars
        if retrieved:
            question_tokens = tokens(str(row["prompt"]))
            ranked = sorted(
                retrieved,
                key=lambda item: len(question_tokens & item[2]) / max(1, len(question_tokens)),
                reverse=True,
            )
            chosen_examples = [(question, answer) for question, answer, _ in ranked[: args.retrieve]]
        if chosen_examples:
            prompt = few_shot_prompt(tokenizer, str(row["prompt"]), chosen_examples, system=system)
        else:
            prompt = chat_prompt(tokenizer, str(row["prompt"]), system=system)
        if args.execute:
            text = greedy_with_calculator(model, tokenizer, prompt, args.max_tokens)
        elif args.samples <= 1:
            text = generate(model, tokenizer, prompt=prompt, max_tokens=args.max_tokens, verbose=False)
        else:
            draws = []
            for _ in range(args.samples):
                draw = one_completion(prompt, args.temperature)
                if args.tools:
                    draw = cut_repeated_tail(draw)
                draws.append(draw)
            counts = Counter(answer_key(draw) for draw in draws if answer_key(draw))
            chosen = counts.most_common(1)[0][0] if counts else ""
            text = next((draw for draw in draws if answer_key(draw) == chosen), draws[0])
        if args.tools:
            text = cut_repeated_tail(text)
        text_ok = exact_match(text, str(row["gold"]))
        tool_ok = tool_match(text, str(row["gold"])) if args.tools else text_ok
        f = format_box(text)
        b = boxed_match(text, str(row["gold"]))
        n_calls = tool_calls(text) if args.tools else 0
        expr = expression_f1(text, str(row["solution"]), str(row["prompt"])) if args.tools else 0.0
        hits += text_ok
        fmt += f
        boxed_hits += b
        tool_hits += tool_ok
        expr_sum += expr
        calls += 1 if n_calls else 0
        records.append(
            {
                "task_id": row["task_id"],
                "gold": str(row["gold"]),
                "correct": tool_ok if args.tools else text_ok,
                "text_correct": text_ok,
                "format": f,
                "boxed": b,
                "tool_calls": n_calls,
                "expression_f1": expr,
                "completion": text,
            }
        )
        shown = tool_ok if args.tools else text_ok
        print(f"{i+1}/{len(df)} acc={tool_hits/(i+1) if args.tools else hits/(i+1):.3f} gold={row['gold']} ok={shown}", flush=True)

    summary = {
        "dataset": "openai/gsm8k",
        "config": "main",
        "split": args.split,
        "offset": args.offset,
        "n": len(df),
        "model": args.model,
        "adapter": args.adapter or None,
        "protocol": (
            f"{args.shots}-shot, calculator executed during decoding"
            if args.execute
            else f"{args.shots}-shot, calculator tool, executed calc() value"
            if args.tools
            else f"{args.shots}-shot, answer in <answer>, exact number"
        ),
        "shots": args.shots,
        "retrieve": args.retrieve,
        "checklist": args.checklist,
        "execute": args.execute,
        "tools": args.tools,
        "max_tokens": args.max_tokens,
        "samples": args.samples,
        "temperature": args.temperature if args.samples > 1 else 0.0,
        "exact_match": hits / len(df),
        "tool_match": tool_hits / len(df),
        "format_rate": fmt / len(df),
        "boxed_match": boxed_hits / len(df),
        "tool_call_rate": calls / len(df),
        "expression_f1": expr_sum / len(df),
        "hack_rate": (
            sum(1 for rec in records if rec["tool_calls"] and rec["correct"] == 1.0 and rec["expression_f1"] == 0.0)
            / len(df)
            if args.tools
            else 0.0
        ),
        "published_reference": {
            "value": 0.444,
            "protocol": "Llama-3.2-1B-Instruct GSM8K 8-shot CoT maj@1, full test",
            "url": "https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct",
        },
        "dataset_url": "https://huggingface.co/datasets/openai/gsm8k",
        "records": records,
    }
    out = ROOT / "results" / "baselines" / f"{args.tag}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {k: summary[k] for k in ("n", "exact_match", "tool_match", "tool_call_rate", "expression_f1", "hack_rate")},
            indent=2,
        )
    )
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
