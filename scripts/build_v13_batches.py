#!/usr/bin/env python3
"""Train-only v13 probes from the v12 failure filters. Never reads the test split.

v13a trains gold number literals the predicted first formula does not already
contain, on right-operation/wrong-number misses.
v13b trains the second formula only when every literal is the previous result
or a literal from the first formula.
v13c trains the operator character when the question already names that
operation and the numbers already match.
Solved replay rows, when a batch is kept, use the full trace.
A batch is not written when fewer than eight targeted rows survive.
"""

from __future__ import annotations

import ast
import json
import re
import sys
from collections import Counter
from pathlib import Path

import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "environments" / "gsm8k_v1"))

from gsm8k_v1.tools import (  # noqa: E402
    format_value,
    normalize_number,
    ordered_prefix_len,
    predicted_formulas,
    reference_expressions,
    safe_calc,
    tool_trace,
)

OUT = ROOT / "results" / "probes"
MIN_TARGETED = 8
NUM = re.compile(r"(?<![\d.])(\d+(?:\.\d+)?)(?![\d.])")
OP_NAMES = {ast.Add: "+", ast.Sub: "-", ast.Mult: "*", ast.Div: "/"}
CUES = (
    ("/", ("half", "split equally", "divided")),
    ("*", ("twice", "double", "times", "product")),
    ("-", ("left", "remain", "fewer", "shorter", "difference")),
    ("+", ("altogether", "combined", "in total", "sum")),
)


def leaves(expr: str) -> tuple[tuple[str, ...], tuple[str, ...]]:
    tree = ast.parse(expr, mode="eval")
    nums: list[str] = []
    ops: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
            nums.append(str(node.value))
        elif isinstance(node, ast.BinOp):
            ops.append(OP_NAMES.get(type(node.op), "?"))
    return tuple(sorted(nums)), tuple(sorted(ops))


def cell(pred: str, ref: str) -> str:
    pred_nums, pred_ops = leaves(pred)
    ref_nums, ref_ops = leaves(ref)
    if pred_ops == ref_ops and pred_nums != ref_nums:
        return "right-op-wrong-numbers"
    if pred_nums == ref_nums and pred_ops != ref_ops:
        return "right-numbers-wrong-op"
    if pred_nums != ref_nums and pred_ops != ref_ops:
        return "wrong-op-and-wrong-numbers"
    return "other"


def classify(record: dict, task: dict) -> tuple[str, int]:
    if record["correct"] == 1:
        return "solved", 0
    refs = reference_expressions(task["solution"], record["gold"])
    preds = predicted_formulas(record["completion"])
    if not refs or not preds:
        return "other", 0
    prefix = ordered_prefix_len(record["completion"], task["solution"], record["gold"])
    if prefix == 0:
        return "first:" + cell(preds[0], refs[0]), prefix
    if prefix >= len(refs) or prefix >= len(preds):
        return "other", prefix
    return "next:" + cell(preds[prefix], refs[prefix]), prefix


def raw_formulas(traced: str) -> list[str]:
    found: list[str] = []
    key = "calc("
    i = 0
    while True:
        start = traced.find(key, i)
        if start < 0:
            break
        depth = 1
        k = start + len(key)
        while k < len(traced) and depth:
            if traced[k] == "(":
                depth += 1
            elif traced[k] == ")":
                depth -= 1
            k += 1
        if depth != 0:
            break
        expr = traced[start + len(key) : k - 1]
        try:
            safe_calc(expr)
        except (SyntaxError, ValueError, ZeroDivisionError, OverflowError):
            i = k
            continue
        found.append(expr)
        i = k
    return found


def number_focus(raw_expr: str, predicted: str) -> str | None:
    pred_left: Counter[str] = Counter(normalize_number(n) for n in NUM.findall(predicted))
    pieces: list[str] = []
    for lit in NUM.findall(raw_expr):
        key = normalize_number(lit)
        if not key:
            continue
        if pred_left[key] > 0:
            pred_left[key] -= 1
            continue
        pieces.append(lit)
    if not pieces:
        return None
    return "|".join(pieces)


def cued_operator(question: str) -> str | None:
    text = question.lower()
    found: list[str] = []
    for op, words in CUES:
        for word in words:
            if re.search(rf"\b{re.escape(word)}\b", text):
                found.append(op)
                break
    if len(found) != 1:
        return None
    return found[0]


def binary_ops(expr: str, op: str) -> int:
    count = 0
    i = 0
    while True:
        at = expr.find(op, i)
        if at < 0:
            return count
        prev = expr[at - 1] if at else ""
        if prev in "0123456789)":
            count += 1
        i = at + 1


def uses_only_previous(first: str, second: str) -> bool:
    try:
        previous = normalize_number(format_value(safe_calc(first)))
    except (SyntaxError, ValueError, ZeroDivisionError, OverflowError):
        return False
    allowed = {normalize_number(n) for n in NUM.findall(first)}
    allowed.discard("")
    literals = [normalize_number(n) for n in NUM.findall(second)]
    if not previous or not literals or previous not in literals:
        return False
    return all(lit and (lit == previous or lit in allowed) for lit in literals)


def row_payload(task: dict, loss_span: str, focus: str = "") -> dict:
    payload = {
        "task_id": task["task_id"],
        "split": "train",
        "prompt": task["prompt"],
        "solution": task["solution"],
        "gold": str(task["gold"]),
        "loss_span": loss_span,
    }
    if focus:
        payload["focus"] = focus
    return payload


def write_batch(name: str, targeted: list[dict], replay: list[dict]) -> None:
    path = OUT / name
    with path.open("w", encoding="utf-8") as handle:
        for item in targeted + replay:
            handle.write(json.dumps(item) + "\n")
    print(f"wrote {path} targeted {len(targeted)} replay {len(replay)}", flush=True)


def main() -> None:
    manifest = json.loads((OUT / "v12_manifest.json").read_text())
    banned = {
        task_id
        for key in ("number_probe", "both_probe", "next_probe", "op_probe", "control")
        for task_id in manifest[key]
    }
    replay_ids = list(manifest["replay"])
    baseline = json.loads((ROOT / "results" / "baselines" / "v7_train800.json").read_text())
    records = {row["task_id"]: row for row in baseline["records"]}
    tasks = {
        row["task_id"]: row
        for row in pq.read_table(
            ROOT / "data" / "processed" / "tasks.parquet",
            columns=["task_id", "split", "prompt", "solution", "gold"],
        ).to_pylist()
        if row["split"] == "train" and row["task_id"] in records
    }
    if any(task_id not in tasks for task_id in records):
        raise SystemExit("v7 train baseline has a row missing from the train split")

    numbers: list[dict] = []
    nxt: list[dict] = []
    ops: list[dict] = []
    seen_replay: list[dict] = []
    counts: Counter[str] = Counter()

    for task_id, record in records.items():
        task = tasks[task_id]
        label, prefix = classify(record, task)
        counts[label] += 1
        traced = tool_trace(str(task["solution"]), str(task["gold"]))
        if task_id in replay_ids and traced is not None:
            seen_replay.append(row_payload(task, "all"))
        if task_id in banned or traced is None:
            continue
        formulas = raw_formulas(traced)
        if not formulas:
            continue
        if label == "first:right-op-wrong-numbers":
            preds = predicted_formulas(record["completion"])
            focus = number_focus(formulas[0], preds[0]) if preds else None
            if focus:
                numbers.append(row_payload(task, "numbers-only", focus))
        elif label.startswith("next:") and prefix == 1 and len(formulas) >= 2:
            if uses_only_previous(formulas[0], formulas[1]):
                nxt.append(row_payload(task, "next-only"))
        elif label == "first:right-numbers-wrong-op":
            op = cued_operator(str(task["prompt"]))
            if op is None:
                continue
            _, gold_ops = leaves(reference_expressions(task["solution"], str(task["gold"]))[0])
            if not gold_ops or any(item != op for item in gold_ops):
                continue
            count = binary_ops(formulas[0], op)
            if count != len(gold_ops):
                continue
            ops.append(row_payload(task, "operator-only", "|".join([op] * count)))

    replay = [item for item in seen_replay if item["task_id"] in set(replay_ids)]
    # Preserve the manifest order and drop a replay row that did not trace.
    replay_by_id = {item["task_id"]: item for item in replay}
    replay = [replay_by_id[task_id] for task_id in replay_ids if task_id in replay_by_id]

    report = {
        "min_targeted": MIN_TARGETED,
        "bucket_counts": dict(counts),
        "replay_kept": [item["task_id"] for item in replay],
        "numbers_targeted": [item["task_id"] for item in numbers],
        "next_targeted": [item["task_id"] for item in nxt],
        "operator_targeted": [item["task_id"] for item in ops],
        "trained": {},
    }
    if len(numbers) >= MIN_TARGETED:
        write_batch("v13a_numbers.jsonl", numbers, replay)
        report["trained"]["v13a"] = len(numbers) + len(replay)
    else:
        print(f"skip v13a numbers targeted {len(numbers)}", flush=True)
    if len(nxt) >= MIN_TARGETED:
        write_batch("v13b_next.jsonl", nxt, replay)
        report["trained"]["v13b"] = len(nxt) + len(replay)
    else:
        print(f"skip v13b next targeted {len(nxt)}", flush=True)
    if len(ops) >= MIN_TARGETED:
        write_batch("v13c_operator.jsonl", ops, replay)
        report["trained"]["v13c"] = len(ops) + len(replay)
    else:
        print(f"skip v13c operator targeted {len(ops)}", flush=True)
    (OUT / "v13_manifest.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"wrote {OUT / 'v13_manifest.json'}", flush=True)


if __name__ == "__main__":
    main()
