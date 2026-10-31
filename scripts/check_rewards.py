#!/usr/bin/env python3
"""Gold GSM8K completions must score 1. Garbage and wrong numbers must not."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "environments" / "gsm8k_v1"))

from gsm8k_v1.rewards import combined, gold_from_solution, score_task  # noqa: E402

TASKS = ROOT / "data" / "processed" / "tasks.parquet"


def main() -> None:
    solution = "Janet has 16. 16-3=13.\n#### 13"
    gold = gold_from_solution(solution)
    cases = [
        ("<answer>13</answer>", gold, 1.0),
        ("The result is #### 13", gold, 1.0),
        ("<answer>13.0</answer>", gold, 1.0),
        ("<answer>1,000</answer>", "1000", 1.0),
        ("<answer>14</answer>", gold, 0.0),
        ("", gold, 0.0),
        ("I don't know", gold, 0.0),
        ("The meeting is at 5:00", "0", 0.0),
        ("The meeting is at 5:00", "5", 0.0),
    ]
    failed = 0
    for comp, g, want in cases:
        got = score_task(comp, g)
        if got != want:
            failed += 1
            print(f"FAIL {comp!r} gold={g!r} got {got} want {want}")
    fmt_only = combined("<answer>nope</answer>", gold, format_weight=0.1)
    if fmt_only > 0.15:
        failed += 1
        print("FAIL format weight", fmt_only)

    if TASKS.exists():
        import pandas as pd

        df = pd.read_parquet(TASKS)
        df = df[df["split"] == "test"].head(50)
        for _, row in df.iterrows():
            wrapped = f"<answer>{row['gold']}</answer>"
            if score_task(wrapped, row["gold"]) < 0.99:
                failed += 1
                print("FAIL replay", row["task_id"], row["gold"])
    failed += _tool_checks()
    if failed:
        raise SystemExit(f"{failed} reward checks failed")
    print("reward checks passed")


def _close(got: float, want: float, tol: float = 1e-6) -> bool:
    return abs(got - want) <= tol


def _tool_checks() -> int:
    import json

    import pandas as pd

    from gsm8k_v1.tools import (
        calculator_insertion,
        canonicalize,
        expression_f1,
        reference_expressions,
        tool_match,
        tool_reward,
        tool_trace,
    )

    failed = 0
    df = pd.read_parquet(TASKS)
    train = df[df["split"] == "train"].set_index("task_id")
    natalia = train.loc["gsm8k:train:0"]
    solution, gold = str(natalia["solution"]), str(natalia["gold"])

    def reward(text: str, row=natalia) -> float:
        return tool_reward(text, str(row["gold"]), str(row["solution"]), str(row["prompt"]))

    cases = {
        "trace": "calc(48/2) calc(48+24)",
        "parens": "calc((48/2)) calc((48+24))",
        "commute": "calc(48/2) calc(24+48)",
        "spaces": "calc(48 / 2) calc(48 + 24)",
        "decimal": "calc(10*0.5)",
    }
    if not _close(reward(cases["trace"]), 1.0):
        failed += 1
        print("FAIL natalia trace", reward(cases["trace"]))
    if not _close(reward(cases["parens"]), 1.0):
        failed += 1
        print("FAIL parens", reward(cases["parens"]))
    if not _close(reward(cases["commute"]), 1.0):
        failed += 1
        print("FAIL commute", reward(cases["commute"]))
    if not _close(reward(cases["spaces"]), 1.0):
        failed += 1
        print("FAIL spaces", reward(cases["spaces"]))
    if reward("calc(72)") != 0.0 or tool_match("calc(72)", gold) != 0.0:
        failed += 1
        print("FAIL bare", reward("calc(72)"))
    if tool_match("calc(72*1)", gold) != 0.0 or tool_match("calc(+72)", gold) != 0.0:
        failed += 1
        print("FAIL identity match")
    partial = reward("calc(48/2)")
    identity = reward("calc(72*1)")
    if not identity < partial:
        failed += 1
        print("FAIL identity vs partial", identity, partial)
    plus_junk = reward("calc(48/2) calc(48+24) calc(72)")
    if not plus_junk < 1.0 or tool_match("calc(48/2) calc(48+24) calc(72)", gold) != 1.0:
        failed += 1
        print("FAIL extra bare", plus_junk)
    four = "<<1+1=2>> <<2+2=4>> <<4+1=5>> <<5+5=10>>\n#### 10"
    only_final = tool_reward("calc(5+5)", "10", four)
    padded = tool_reward("calc(5+5) calc(1) calc(2) calc(3) calc(4)", "10", four)
    if not padded < only_final:
        failed += 1
        print("FAIL padding outranks short formula", padded, only_final)
    one = "<<30+40=70>>70\n#### 70"
    perfect_one = tool_reward("calc(30+40)", "70", one)
    piled = tool_reward(" ".join(["calc(10+60)"] * 10), "70", one)
    if not piled < perfect_one or perfect_one < 0.99:
        failed += 1
        print("FAIL one-step pile", piled, perfect_one)
    wrong = reward("calc(20+4) calc(70+2)")
    if expression_f1("calc(20+4) calc(70+2)", solution, "") != 0.0 or not wrong < 1.0:
        failed += 1
        print("FAIL wrong formula", wrong)
    if canonicalize("10*0.5") != canonicalize("10*.5"):
        failed += 1
        print("FAIL decimal canon", canonicalize("10*0.5"), canonicalize("10*.5"))
    if canonicalize("0.025*5000") != canonicalize("5000*(2.5/100)"):
        failed += 1
        print("FAIL percent canon", canonicalize("0.025*5000"), canonicalize("5000*(2.5/100)"))
    if canonicalize("(20*20)/100") == canonicalize("0.2*20"):
        failed += 1
        print("FAIL percent collapsed a real division")
    if canonicalize("20+4") == canonicalize("48/2"):
        failed += 1
        print("FAIL numeric collapse")
    if calculator_insertion("steps calc(48/2)") != " = 24":
        failed += 1
        print("FAIL insert", calculator_insertion("steps calc(48/2)"))
    if calculator_insertion("calc((48/2))") != " = 24":
        failed += 1
        print("FAIL insert paren", calculator_insertion("calc((48/2))"))
    if calculator_insertion("calc(48/2) = 24") is not None:
        failed += 1
        print("FAIL insert already valued")

    trace55 = tool_trace(str(train.loc["gsm8k:train:55"]["solution"]), "75")
    if trace55 is None or "7575" in trace55 or not _close(reward(trace55, train.loc["gsm8k:train:55"]), 1.0):
        failed += 1
        print("FAIL train55 trace", None if trace55 is None else trace55[-180:])
    trace4188 = tool_trace(str(train.loc["gsm8k:train:4188"]["solution"]), str(train.loc["gsm8k:train:4188"]["gold"]))
    if trace4188 is None or "1212" in trace4188:
        failed += 1
        print("FAIL train4188 glue", trace4188)

    kept = 0
    bad_last = 0
    from gsm8k_v1.tools import (
        answer_formula_span,
        format_value,
        ordered_prefix_len,
        safe_calc,
    )
    from gsm8k_v1.rewards import normalize_number

    test = df[df["split"] == "test"].reset_index(drop=True)
    vacuum = test.iloc[13]
    discount = test.iloc[24]
    profit = test.iloc[15]
    for label, row in (("vacuum", vacuum), ("discount", discount), ("profit", profit)):
        refs = reference_expressions(str(row["solution"]), str(row["gold"]))
        if not refs or normalize_number(format_value(safe_calc(refs[-1]))) != normalize_number(str(row["gold"])):
            failed += 1
            print("FAIL recovered reference", label, refs)
    if ordered_prefix_len("calc(48/2)", solution, gold) != 1:
        failed += 1
        print("FAIL prefix", ordered_prefix_len("calc(48/2)", solution, gold))
    if not reward("calc(48/2)") > reward("calc(2*2)"):
        failed += 1
        print("FAIL prefix reward", reward("calc(48/2)"), reward("calc(2*2)"))
    if not reward("calc(48/2)") < reward("calc(48/2) calc(48+24)"):
        failed += 1
        print("FAIL unfinished below finished")
    continued = "calc(0.025*5000) calc(0.012*8000) calc(125+96)"
    if answer_formula_span(continued, str(profit["solution"]), str(profit["gold"])) is None:
        failed += 1
        print("FAIL answer span", continued)
    if tool_match(continued, str(profit["gold"])) != 0.0:
        failed += 1
        print("FAIL continued profit still counted")

    for task_id, row in train.iterrows():
        refs = reference_expressions(str(row["solution"]), str(row["gold"]))
        if not refs:
            continue
        kept += 1
        last = normalize_number(format_value(safe_calc(refs[-1])))
        if last != normalize_number(str(row["gold"])):
            bad_last += 1
            if bad_last <= 3:
                print("FAIL reference", task_id, refs[-1], last, row["gold"])
    if bad_last:
        failed += bad_last
    print(f"reference rows {kept} of {len(train)}")

    out = TASKS.parents[0] / "train_expressions.jsonl"
    with out.open("w", encoding="utf-8") as handle:
        for task_id, row in train.iterrows():
            refs = reference_expressions(str(row["solution"]), str(row["gold"]))
            if not refs:
                continue
            handle.write(
                json.dumps({"task_id": task_id, "gold": str(row["gold"]), "expressions": refs})
                + "\n"
            )

    saved = TASKS.parents[1].parent / "results" / "baselines" / "expr_1.json"
    if saved.exists():
        doc = json.loads(saved.read_text())
        test = df[df["split"] == "test"].set_index("task_id")
        match_sum = 0.0
        expr_sum = 0.0
        hack = 0
        for rec in doc["records"]:
            row = test.loc[rec["task_id"]]
            match_sum += tool_match(rec["completion"], str(row["gold"]))
            overlap = expression_f1(rec["completion"], str(row["solution"]), str(row["prompt"]))
            expr_sum += overlap
            if tool_match(rec["completion"], str(row["gold"])) == 1.0 and overlap == 0.0:
                hack += 1
        n = len(doc["records"])
        print(
            f"rescored expr_1 n={n} tool_match={match_sum / n:.3f} "
            f"expression_f1={expr_sum / n:.3f} hack_hits={hack}"
        )

    import math

    def k3(logprob: float, ref: float) -> float:
        ratio = logprob - ref
        return math.exp(-ratio) + ratio - 1.0

    if k3(0.2, 0.2) > 1e-12:
        failed += 1
        print("FAIL k3 zero")
    # At equality the penalty does not move. A small positive advantage still
    # increases the sampled log-prob: slope of -adv*lp + beta*k3 is about -adv.
    advantage = 0.01
    beta = 0.05
    base = 0.2
    eps = 1e-3

    def objective(logprob: float) -> float:
        return (-advantage) * logprob + beta * k3(logprob, base)

    slope = (objective(base + eps) - objective(base - eps)) / (2 * eps)
    if slope >= 0:
        failed += 1
        print("FAIL kl slope", slope)
    return failed


if __name__ == "__main__":
    main()
