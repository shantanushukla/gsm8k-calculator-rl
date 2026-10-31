#!/usr/bin/env python3
"""Small train-only batches for the five formula-error reasons.

The GSM8K test split is never read. Each batch trains on one slice of the
first 800 train rows and lists a disjoint probe of the same error.
"""

from __future__ import annotations

import ast
import json
import random
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
ANN = re.compile(r"<<\s*([^=<>]+?)\s*=\s*([^=<>]+?)\s*>>")
NUM = re.compile(r"(?<![\d.])(\d+(?:\.\d+)?)(?![\d.])")
OPS = {ast.Add: "addition", ast.Sub: "subtraction", ast.Mult: "multiplication", ast.Div: "division"}


def leaves(expr: str) -> tuple[tuple[str, ...], tuple[str, ...]]:
    tree = ast.parse(expr, mode="eval")
    nums: list[str] = []
    ops: list[str] = []
    names = {ast.Add: "+", ast.Sub: "-", ast.Mult: "*", ast.Div: "/"}
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
            nums.append(str(node.value))
        elif isinstance(node, ast.BinOp):
            ops.append(names.get(type(node.op), "?"))
    return tuple(sorted(nums)), tuple(sorted(ops))


def cell(pred: str, ref: str) -> str:
    pred_nums, pred_ops = leaves(pred)
    ref_nums, ref_ops = leaves(ref)
    same_nums = pred_nums == ref_nums
    same_ops = pred_ops == ref_ops
    if same_ops and not same_nums:
        return "right-op-wrong-numbers"
    if same_nums and not same_ops:
        return "right-numbers-wrong-op"
    if not same_nums and not same_ops:
        return "wrong-op-and-wrong-numbers"
    return "other"


def bucket(record: dict, task: dict) -> str:
    if record["correct"] == 1:
        return "solved"
    refs = reference_expressions(task["solution"], record["gold"])
    preds = predicted_formulas(record["completion"])
    if not refs or not preds:
        return "other"
    prefix = ordered_prefix_len(record["completion"], task["solution"], record["gold"])
    if prefix == 0:
        return "first:" + cell(preds[0], refs[0])
    if prefix >= len(refs) or prefix >= len(preds):
        return "other"
    return "next:" + cell(preds[prefix], refs[prefix])


def outermost_op(expr: str) -> str:
    tree = ast.parse(expr.replace("^", "**"), mode="eval").body
    if isinstance(tree, ast.BinOp):
        return OPS.get(type(tree.op), "arithmetic")
    return "arithmetic"


def replace_numbers(text: str, mapping: dict[str, str]) -> str:
    def repl(match: re.Match[str]) -> str:
        return mapping.get(match.group(1), match.group(1))

    return NUM.sub(repl, text)


def replace_outside_annotations(text: str, mapping: dict[str, str]) -> str:
    parts: list[str] = []
    last = 0
    for match in ANN.finditer(text):
        parts.append(replace_numbers(text[last : match.start()], mapping))
        parts.append(match.group(0))
        last = match.end()
    parts.append(replace_numbers(text[last:], mapping))
    return "".join(parts)


def perturb(prompt: str, solution: str, gold: str) -> dict | None:
    """Change one question number and recompute the annotated formulas."""
    anns = ANN.findall(solution)
    if not anns:
        return None
    prompt_nums = NUM.findall(prompt)
    first_lits = set(NUM.findall(anns[0][0]))
    candidates = [int(n) for n in prompt_nums if n.isdigit() and int(n) >= 2 and n in first_lits]
    if not candidates:
        return None
    old = str(max(candidates))
    occupied = set(prompt_nums) | set(NUM.findall(solution))
    new = None
    for delta in (1, 3, 4, 5):
        guess = str(int(old) + delta)
        if guess not in occupied:
            new = guess
            break
    if new is None:
        return None
    question_map = {old: new}
    produced: dict[str, str] = {}
    rebuilt: list[tuple[str, str, str]] = []
    for expr, shown in anns:
        rewritten = replace_numbers(expr, {**question_map, **produced})
        try:
            value = normalize_number(format_value(safe_calc(rewritten)))
            old_value = normalize_number(format_value(safe_calc(expr)))
        except (SyntaxError, ValueError, ZeroDivisionError, OverflowError):
            return None
        if value == old_value:
            return None
        rebuilt.append((expr, shown, f"<<{rewritten}={value}>>", value))
        produced[normalize_number(shown)] = value
        produced[old_value] = value
    new_gold = rebuilt[-1][3]
    if new_gold == normalize_number(str(gold)):
        return None
    prompt_left = set(prompt_nums) - {old}
    if any(key in prompt_left for key in produced):
        return None
    text = solution
    for expr, shown, block, _value in rebuilt:
        source = f"<<{expr}={shown}>>"
        if source not in text:
            return None
        text = text.replace(source, block, 1)
    text = replace_outside_annotations(text, {**question_map, **produced})
    text = re.sub(r"####.*", f"#### {new_gold}", text)
    new_prompt = replace_numbers(prompt, question_map)
    if old in NUM.findall(new_prompt) or new_gold == normalize_number(str(gold)):
        return None
    return {"prompt": new_prompt, "solution": text, "gold": new_gold}


def operation_sentence(solution: str) -> str | None:
    match = ANN.search(solution)
    if match is None:
        return None
    try:
        name = outermost_op(match.group(1))
    except (SyntaxError, ValueError):
        return None
    sentence = f"The operation is {name}. "
    line_start = solution.rfind("\n", 0, match.start()) + 1
    return solution[:line_start] + sentence + solution[line_start:]


def write_ids(path: Path, ids: list[str]) -> None:
    path.write_text("\n".join(ids) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def take(items: list[str], n: int, rng: random.Random) -> tuple[list[str], list[str]]:
    pool = items[:]
    rng.shuffle(pool)
    probe = pool[:n]
    train = pool[n:]
    return train, probe


def main() -> None:
    tasks = {
        row["task_id"]: row
        for row in pq.read_table(ROOT / "data" / "processed" / "tasks.parquet").to_pylist()
        if row["split"] == "train"
    }
    records = json.load(open(ROOT / "results" / "baselines" / "v7_train800.json"))["records"]
    groups: dict[str, list[str]] = {}
    for record in records:
        task = tasks[record["task_id"]]
        groups.setdefault(bucket(record, task), []).append(record["task_id"])
    print("train800 buckets")
    for name, ids in sorted(groups.items(), key=lambda item: -len(item[1])):
        print(f"{len(ids):4d}  {name}")

    rng = random.Random(21)
    solved = groups.get("solved", [])
    wrong_numbers = groups.get("first:right-op-wrong-numbers", [])
    wrong_both = groups.get("first:wrong-op-and-wrong-numbers", [])
    wrong_op = groups.get("first:right-numbers-wrong-op", [])
    continuation = groups.get("next:wrong-op-and-wrong-numbers", []) + groups.get(
        "next:right-op-wrong-numbers", []
    )
    _, number_probe = take(wrong_numbers, 12, rng)
    number_train_ids = [task_id for task_id in wrong_numbers if task_id not in set(number_probe)]
    _, both_probe = take(wrong_both, 12, rng)
    _, op_probe = take(wrong_op, min(12, max(0, len(wrong_op) // 3)), rng)
    op_train_ids = [task_id for task_id in wrong_op if task_id not in set(op_probe)]
    _, next_probe = take(continuation, 12, rng)
    next_train_ids = [task_id for task_id in continuation if task_id not in set(next_probe)]

    used: set[str] = set(number_probe) | set(both_probe) | set(op_probe) | set(next_probe)
    solved_free = [task_id for task_id in solved if task_id not in used]
    rng.shuffle(solved_free)
    control = solved_free[:8]
    replay = solved_free[8:24]
    perturb_source = solved_free[24:]

    OUT.mkdir(parents=True, exist_ok=True)
    perturbed = []
    for task_id in perturb_source:
        task = tasks[task_id]
        changed = perturb(task["prompt"], task["solution"], str(task["gold"]))
        if changed is None or tool_trace(changed["solution"], changed["gold"]) is None:
            continue
        perturbed.append(
            {
                "task_id": task_id + ":num",
                "split": "train",
                "prompt": changed["prompt"],
                "solution": changed["solution"],
                "gold": changed["gold"],
                "loss_span": "all",
                "source_task_id": task_id,
            }
        )
        if len(perturbed) >= 16:
            break
    if len(perturbed) < 8:
        raise SystemExit(f"only {len(perturbed)} number-changed rows")

    number_rows = perturbed + [
        {
            "task_id": task_id,
            "split": "train",
            "prompt": tasks[task_id]["prompt"],
            "solution": tasks[task_id]["solution"],
            "gold": str(tasks[task_id]["gold"]),
            "loss_span": "all",
        }
        for task_id in replay
    ]
    next_rows = []
    for task_id in next_train_ids[:16]:
        task = tasks[task_id]
        next_rows.append(
            {
                "task_id": task_id,
                "split": "train",
                "prompt": task["prompt"],
                "solution": task["solution"],
                "gold": str(task["gold"]),
                "loss_span": "next-only",
            }
        )
    next_rows.extend(
        {
            "task_id": task_id,
            "split": "train",
            "prompt": tasks[task_id]["prompt"],
            "solution": tasks[task_id]["solution"],
            "gold": str(tasks[task_id]["gold"]),
            "loss_span": "all",
        }
        for task_id in replay
    )
    op_rows = []
    for task_id in op_train_ids[:16]:
        task = tasks[task_id]
        solution = operation_sentence(task["solution"])
        if solution is None:
            continue
        op_rows.append(
            {
                "task_id": task_id,
                "split": "train",
                "prompt": task["prompt"],
                "solution": solution,
                "gold": str(task["gold"]),
                "loss_span": "op-lead",
            }
        )
    op_rows.extend(
        {
            "task_id": task_id,
            "split": "train",
            "prompt": tasks[task_id]["prompt"],
            "solution": tasks[task_id]["solution"],
            "gold": str(tasks[task_id]["gold"]),
            "loss_span": "all",
        }
        for task_id in replay
    )

    write_jsonl(OUT / "v12a_numbers.jsonl", number_rows)
    write_jsonl(OUT / "v12b_next.jsonl", next_rows)
    write_jsonl(OUT / "v12c_op.jsonl", op_rows)
    write_ids(OUT / "v12a_probe_ids.txt", number_probe + control)
    write_ids(OUT / "v12a_both_probe_ids.txt", both_probe)
    write_ids(OUT / "v12b_probe_ids.txt", next_probe + control)
    write_ids(OUT / "v12c_probe_ids.txt", op_probe + control)
    write_ids(OUT / "v12_control_ids.txt", control)
    write_ids(OUT / "v12d_source_ids.txt", [task_id for task_id in wrong_both if task_id not in set(both_probe)][:16])
    manifest = {
        "number_rows": len(number_rows),
        "perturbed": len(perturbed),
        "next_rows": len(next_rows),
        "next_targeted": sum(1 for row in next_rows if row["loss_span"] == "next-only"),
        "op_rows": len(op_rows),
        "op_targeted": sum(1 for row in op_rows if row["loss_span"] == "op-lead"),
        "number_probe": number_probe,
        "both_probe": both_probe,
        "next_probe": next_probe,
        "op_probe": op_probe,
        "control": control,
        "replay": replay,
    }
    (OUT / "v12_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in manifest.items() if not isinstance(value, list)}, indent=2))
    print("--- example number change ---")
    print(perturbed[0]["source_task_id"])
    print(perturbed[0]["prompt"][:400])
    print(perturbed[0]["solution"][:500])
    print("gold", perturbed[0]["gold"])
    if op_rows:
        print("--- example operation sentence ---")
        print(op_rows[0]["solution"][:400])


if __name__ == "__main__":
    main()
