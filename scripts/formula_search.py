#!/usr/bin/env python3
"""Pick calculator steps from candidates built in code. The test split is only scored.

The model does not invent the formula. Code reads the question numbers, adds a
constant only when the wording licenses it, and asks the frozen model to choose
the next pair and the next operation. Later steps must use the previous result.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "environments" / "gsm8k_v1"))
sys.path.insert(0, str(ROOT / "scripts"))

from gsm8k_v1.tools import (  # noqa: E402
    format_value,
    normalize_number,
    ordered_prefix_len,
    predicted_formulas,
    reference_expressions,
    safe_calc,
)
from model_io import chat_prompt, encode, load_policy  # noqa: E402

_UNITS = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11,
    "twelve": 12, "thirteen": 13, "fourteen": 14, "fifteen": 15, "sixteen": 16,
    "seventeen": 17, "eighteen": 18, "nineteen": 19,
}
_TENS = {
    "twenty": 20, "thirty": 30, "forty": 40, "fifty": 50, "sixty": 60,
    "seventy": 70, "eighty": 80, "ninety": 90,
}
_WORD = "|".join([*_UNITS, *_TENS, "hundred", "thousand"])
NUM = re.compile(
    rf"(?<![\d.])(\d{{1,3}}(?:,\d{{3}})+|\d+(?:\.\d+)?)(?![\d.])"
    rf"|\b({_WORD})(?:-({_WORD}))?\b",
    re.I,
)
LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
CHOOSE = (
    "Choose the correct next calculation for the grade-school math problem. "
    "The calculator will compute it."
)


@dataclass(frozen=True)
class Operand:
    text: str
    source: str  # "question", "constant", "result"


@dataclass(frozen=True)
class Formula:
    expr: str
    value: str
    english: str


def _word_value(head: str, tail: str | None) -> str | None:
    head = head.lower()
    tail = tail.lower() if tail else None
    if head in _UNITS and tail is None:
        return str(_UNITS[head])
    if head in _TENS and (tail is None or tail in _UNITS):
        return str(_TENS[head] + (_UNITS[tail] if tail else 0))
    if head == "hundred" and tail is None:
        return "100"
    if head == "thousand" and tail is None:
        return "1000"
    return None


def question_numbers(question: str) -> list[str]:
    found: list[str] = []
    for match in NUM.finditer(question or ""):
        if match.group(1):
            found.append(match.group(1).replace(",", ""))
            continue
        value = _word_value(match.group(2), match.group(3))
        if value is not None:
            found.append(value)
    return found


def licensed_constants(question: str) -> list[Operand]:
    text = (question or "").lower()
    found: list[str] = []
    if re.search(r"\bhalf\b|\bhalves\b|\btwice\b|\bdouble\b", text):
        found.append("2")
    if re.search(r"\bpercent\b|\bpercentage\b|%", text):
        found.append("100")
    if re.search(r"\bminutes?\b", text):
        found.append("60")
    if re.search(r"\bweeks?\b", text):
        found.append("7")
    if re.search(r"\bdozen\b", text):
        found.append("12")
    if re.search(r"\byears?\b", text) and re.search(r"\bmonths?\b", text):
        found.append("12")
    if re.search(r"\byears?\b", text) and re.search(r"\bdays?\b", text):
        found.append("365")
    if re.search(r"\bdays?\b", text) and re.search(r"\bhours?\b", text):
        found.append("24")
    constants: list[Operand] = []
    seen: set[str] = set()
    for item in found:
        if item in seen:
            continue
        seen.add(item)
        constants.append(Operand(item, "constant"))
    return constants


def _eval(expr: str) -> Formula | None:
    try:
        value = normalize_number(format_value(safe_calc(expr)))
    except (SyntaxError, ValueError, ZeroDivisionError, OverflowError):
        return None
    if not value:
        return None
    return Formula(expr, value, "")


def formulas_for(left: Operand, right: Operand) -> list[Formula]:
    found: list[Formula] = []
    seen: set[str] = set()
    rendered = [
        (f"{left.text}+{right.text}", f"{left.text} plus {right.text}"),
        (f"{left.text}*{right.text}", f"{left.text} times {right.text}"),
        (f"{left.text}-{right.text}", f"{left.text} minus {right.text}"),
        (f"{right.text}-{left.text}", f"{right.text} minus {left.text}"),
        (f"{left.text}/{right.text}", f"{left.text} divided by {right.text}"),
        (f"{right.text}/{left.text}", f"{right.text} divided by {left.text}"),
    ]
    for expr, english in rendered:
        if expr in seen:
            continue
        formula = _eval(expr)
        if formula is None:
            continue
        seen.add(expr)
        found.append(Formula(expr, formula.value, f"{english} = {formula.value}"))
    return found


def operand_pool(question: str, earlier_results: list[str]) -> list[Operand]:
    pool = [Operand(item, "question") for item in question_numbers(question)]
    # A licensed constant stays even when the question already has that number.
    # Half of 2 is the pair 2 and 2, which disappears if the extra 2 is dropped.
    seen = {(item.text, item.source) for item in pool}
    for constant in licensed_constants(question):
        key = (constant.text, constant.source)
        if key in seen:
            continue
        pool.append(constant)
        seen.add(key)
    for result in earlier_results:
        if result in seen:
            continue
        pool.append(Operand(result, "result"))
        seen.add(result)
    return pool


def operand_pairs(pool: list[Operand], require: Operand | None) -> list[tuple[Operand, Operand]]:
    chosen: list[tuple[Operand, Operand]] = []
    if require is None:
        for i, left in enumerate(pool):
            for right in pool[i + 1 :]:
                if left.source == "constant" and right.source == "constant":
                    continue
                chosen.append((left, right))
        return chosen
    for other in pool:
        if other.source == "constant" and require.source == "constant":
            continue
        chosen.append((require, other))
    return chosen


def choice_prompt(tokenizer, question: str, history: list[Formula], lines: list[str]) -> str:
    body = [question.strip()]
    if history:
        done = "\n".join(f"{item.english} via calc({item.expr})" for item in history)
        body.append("Already calculated:\n" + done)
    body.append("Choices:\n" + "\n".join(lines))
    body.append("The correct choice is")
    return chat_prompt(tokenizer, "\n\n".join(body), system=CHOOSE)


def single_token(tokenizer, prompt: str, continuation: str) -> int | None:
    base = encode(tokenizer, prompt)
    full = encode(tokenizer, prompt + continuation)
    if full[: len(base)] != base or len(full) != len(base) + 1:
        return None
    return full[-1]


def pick_index(model, tokenizer, prompt: str, count: int) -> int:
    """Highest next-token probability among the choice letters."""
    scheme = None
    ids: list[int] = []
    for prefix in (" ", "\n"):
        trial = [single_token(tokenizer, prompt, prefix + LETTERS[i]) for i in range(count)]
        if all(token is not None for token in trial) and len(set(trial)) == count:
            scheme = trial
            break
    if scheme is None:
        raise RuntimeError("choice letters do not map to distinct tokens")
    ids = encode(tokenizer, prompt)
    logits = model(mx.array(ids)[None])[0, -1]
    logp = nn.log_softmax(logits.astype(mx.float32), axis=-1)
    mx.eval(logp)
    scores = [float(logp[token]) for token in scheme]
    return max(range(count), key=lambda i: scores[i])


def needs_another(model, tokenizer, question: str, history: list[Formula]) -> bool:
    done = "\n".join(f"{item.english} via calc({item.expr})" for item in history)
    body = (
        f"{question.strip()}\n\nAlready calculated:\n{done}\n\n"
        f"The result so far is {history[-1].value}. Another calculation is needed:"
    )
    prompt = chat_prompt(tokenizer, body, system=CHOOSE)
    yes = single_token(tokenizer, prompt, " Yes")
    no = single_token(tokenizer, prompt, " No")
    if yes is None or no is None:
        raise RuntimeError("Yes and No are not single tokens")
    ids = encode(tokenizer, prompt)
    logits = model(mx.array(ids)[None])[0, -1]
    logp = nn.log_softmax(logits.astype(mx.float32), axis=-1)
    mx.eval(logp)
    return float(logp[yes]) >= float(logp[no])


def choose(model, tokenizer, question: str, history: list[Formula], lines: list[str]) -> int:
    if len(lines) == 1:
        return 0
    if len(lines) > len(LETTERS):
        raise RuntimeError(f"{len(lines)} choices exceed the letter list")
    prompt = choice_prompt(tokenizer, question, history, lines)
    return pick_index(model, tokenizer, prompt, len(lines))


def solve_question(model, tokenizer, question: str, max_steps: int = 4) -> list[Formula]:
    history: list[Formula] = []
    results: list[str] = []
    for _step in range(max_steps):
        if history and not needs_another(model, tokenizer, question, history):
            break
        require = Operand(results[-1], "result") if results else None
        pool = operand_pool(question, results[:-1])
        options = operand_pairs(pool, require)
        if len(options) > len(LETTERS):
            options = [pair for pair in options if "question" in (pair[0].source, pair[1].source)]
            options = options[: len(LETTERS)]
        if not options:
            break
        if len(options) > 1:
            lines = [f"{LETTERS[i]}. {left.text} and {right.text}" for i, (left, right) in enumerate(options)]
            left, right = options[choose(model, tokenizer, question, history, lines)]
        else:
            left, right = options[0]
        formulas = formulas_for(left, right)
        if not formulas:
            break
        lines = [f"{LETTERS[i]}. {formula.english}" for i, formula in enumerate(formulas)]
        picked = formulas[choose(model, tokenizer, question, history, lines)]
        history.append(picked)
        results.append(picked.value)
    return history


def classify(record: dict, task: dict) -> str:
    if record.get("correct") == 1:
        return "v7-correct"
    refs = reference_expressions(task["solution"], str(record["gold"]))
    preds = predicted_formulas(record.get("completion") or "")
    if not refs or not preds:
        return "no-formula"
    prefix = ordered_prefix_len(record["completion"], task["solution"], str(record["gold"]))
    if prefix == 0:
        return "wrong-first"
    if prefix < len(refs):
        return "continuation"
    return "other"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="test", choices=("test", "train"))
    parser.add_argument("--n", type=int, default=200)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--tag", default="formula_search_200")
    parser.add_argument("--show", type=int, default=0)
    args = parser.parse_args()

    import pyarrow.parquet as pq

    rows = [
        row
        for row in pq.read_table(
            ROOT / "data" / "processed" / "tasks.parquet",
            columns=["task_id", "split", "prompt", "solution", "gold"],
        ).to_pylist()
        if row["split"] == args.split
    ]
    rows = rows[args.offset : args.offset + args.n]
    if not rows:
        raise SystemExit("no rows")
    print("loading mlx-community/Llama-3.2-1B-Instruct-4bit", flush=True)
    model, tokenizer = load_policy(
        "mlx-community/Llama-3.2-1B-Instruct-4bit",
        str(ROOT / "results" / "adapters" / "gsm8k-sft-v7"),
    )
    v7_path = ROOT / "results" / "baselines" / "sft_v7_200.json"
    v7 = {}
    if args.split == "test" and args.offset == 0 and v7_path.exists():
        v7 = {row["task_id"]: row for row in json.loads(v7_path.read_text())["records"]}

    hits = 0
    records = []
    buckets = {"wrong-first": [0, 0], "continuation": [0, 0], "no-formula": [0, 0], "other": [0, 0], "v7-correct": [0, 0]}
    for index, row in enumerate(rows, start=1):
        history = solve_question(model, tokenizer, str(row["prompt"]))
        value = history[-1].value if history else ""
        gold = normalize_number(str(row["gold"]))
        correct = 1.0 if value and normalize_number(value) == gold else 0.0
        hits += correct
        kind = "uncompared"
        if row["task_id"] in v7:
            task = {"solution": row["solution"], "gold": row["gold"]}
            kind = classify(v7[row["task_id"]], task)
            buckets[kind][0] += 1
            buckets[kind][1] += correct
        records.append(
            {
                "task_id": row["task_id"],
                "gold": gold,
                "value": value,
                "correct": correct,
                "steps": [item.expr for item in history],
                "v7_bucket": kind,
            }
        )
        mark = "ok" if correct else "miss"
        print(
            f"{index}/{len(rows)} {mark} gold={gold} got={value or '-'} steps={len(history)} acc={hits / index:.3f}",
            flush=True,
        )
        if args.show and index <= args.show:
            for item in history:
                print(f"  {item.expr} = {item.value} ({item.english})", flush=True)

    summary = {
        "split": args.split,
        "n": len(rows),
        "offset": args.offset,
        "model": "mlx-community/Llama-3.2-1B-Instruct-4bit",
        "adapter": "gsm8k_rl/results/adapters/gsm8k-sft-v7",
        "protocol": "code-built formula choices, frozen model picks the letter, previous result is required after step 1",
        "tool_match": hits / len(rows),
        "correct": int(hits),
        "buckets": {name: {"n": counts[0], "correct": int(counts[1])} for name, counts in buckets.items()},
        "records": records,
    }
    out = ROOT / "results" / "baselines" / f"{args.tag}.json"
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({key: summary[key] for key in ("n", "tool_match", "correct", "buckets")}, indent=2), flush=True)
    print(f"wrote {out}", flush=True)


if __name__ == "__main__":
    main()
