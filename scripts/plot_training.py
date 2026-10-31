#!/usr/bin/env python3
"""Write report/figures/*.png and detector JSON from results/metrics.csv."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CSV = ROOT / "results" / "metrics.csv"
FIG = ROOT / "report" / "figures"
DET = ROOT / "results" / "detectors.json"

COLS = [
    "step",
    "reward_mean",
    "reward_std",
    "pct_all_zero_groups",
    "pct_all_one_groups",
    "pct_zero_std_groups",
    "task_score",
    "format_rate",
    "mean_tokens",
    "loss",
]


def _empty_df() -> pd.DataFrame:
    return pd.DataFrame(columns=COLS)


def load() -> pd.DataFrame:
    if not CSV.exists() or CSV.stat().st_size == 0:
        return _empty_df()
    df = pd.read_csv(CSV)
    for c in COLS:
        if c not in df.columns:
            df[c] = pd.NA
    return df


def _save(name: str, title: str, xlabel="step"):
    plt.title(title)
    plt.xlabel(xlabel)
    plt.tight_layout()
    FIG.mkdir(parents=True, exist_ok=True)
    plt.savefig(FIG / name, dpi=140)
    plt.close()


def plot(df: pd.DataFrame) -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    empty = df.empty

    def series(col, ylabel, fname, title):
        plt.figure(figsize=(7, 3.5))
        if empty or col not in df or df[col].dropna().empty:
            plt.text(0.5, 0.5, "no metrics yet", ha="center", va="center", transform=plt.gca().transAxes)
        else:
            plt.plot(df["step"], df[col], marker="o", ms=3)
            plt.ylabel(ylabel)
        _save(fname, title)

    series("loss", "loss", "loss.png", "GRPO loss (policy + KL)")
    if "kl" in df.columns:
        series("kl", "kl", "kl.png", "Mean token KL vs frozen base")
    series("reward_mean", "reward", "reward_mean.png", "Train exact-match rate")
    series("reward_std", "std", "reward_std.png", "Within-group reward std")
    plt.figure(figsize=(7, 3.5))
    if empty:
        plt.text(0.5, 0.5, "no metrics yet", ha="center", va="center", transform=plt.gca().transAxes)
    else:
        plt.plot(df["step"], df["pct_all_zero_groups"], label="all-zero groups")
        plt.plot(df["step"], df["pct_all_one_groups"], label="all-one groups")
        plt.legend()
        plt.ylabel("fraction")
    _save("group_collapse.png", "GRPO group collapse")

    plt.figure(figsize=(7, 3.5))
    if empty:
        plt.text(0.5, 0.5, "no metrics yet", ha="center", va="center", transform=plt.gca().transAxes)
    else:
        plt.plot(df["step"], df["task_score"], label="task score")
        plt.plot(df["step"], df["format_rate"], label="format rate")
        plt.legend()
    _save("task_vs_format.png", "Task vs format (hack detector)")

    series("mean_tokens", "tokens", "length.png", "Mean completion length")

    base_dir = ROOT / "results" / "baselines"
    loaded = {}
    cmp_path = base_dir / "comparison.json"
    if cmp_path.exists():
        loaded.update(json.loads(cmp_path.read_text()))
    order = [
        "base",
        "sft_v3",
        "base_200",
        "sft_v3_200",
        "failed_grpo",
        "sft",
        "grpo",
        "rft",
        "grpo_v3",
        "sft_v4",
        "tag",
        "llama_tool_base",
        "llama_tool_sft",
        "llama_tool_grpo",
        "llama_tool_base_200",
        "llama_tool_best_200",
    ]
    for tag in order:
        path = base_dir / f"{tag}.json"
        if path.exists():
            loaded[tag] = json.loads(path.read_text())["exact_match"]
    labels = [tag for tag in order if tag in loaded]
    labels += [tag for tag in loaded if tag not in labels]
    plt.figure(figsize=(7, 3.5))
    if labels:
        plt.bar(labels, [loaded[tag] for tag in labels])
        plt.ylabel("exact match")
        plt.ylim(0, 1)
    else:
        plt.text(0.5, 0.5, "no baseline comparison yet", ha="center", va="center", transform=plt.gca().transAxes)
    _save("baseline_vs_trained.png", "GSM8K test exact match", xlabel="")


def detect(df: pd.DataFrame) -> dict:
    flags = {
        "always_zero": False,
        "always_one": False,
        "collapse_groups": False,
        "format_hack": False,
        "length_spike": False,
        "eval_drop": False,
    }
    if df.empty or len(df) < 5:
        return {"flags": flags, "notes": "not enough steps"}
    tail = df.tail(20)
    flags["always_zero"] = float(tail["reward_mean"].mean()) < 0.05
    flags["always_one"] = float(tail["reward_mean"].mean()) > 0.95
    flags["collapse_groups"] = float(tail["pct_zero_std_groups"].mean()) > 0.50 if tail["pct_zero_std_groups"].notna().any() else False
    if tail["format_rate"].notna().any() and tail["task_score"].notna().any():
        flags["format_hack"] = float(tail["format_rate"].mean()) > 0.8 and float(tail["task_score"].mean()) < 0.2
    if df["mean_tokens"].notna().sum() > 10:
        flags["length_spike"] = float(df["mean_tokens"].tail(5).mean()) > 2 * float(df["mean_tokens"].head(5).mean())
    cmp_path = ROOT / "results" / "baselines" / "comparison.json"
    notes = ""
    if cmp_path.exists():
        scores = json.loads(cmp_path.read_text())
        base = scores.get("base")
        newer = [
            scores[k]
            for k in ("sft_v3", "rft", "tag", "grpo_v3", "llama_tool_sft", "llama_tool_grpo", "llama_tool_best_200")
            if k in scores
        ]
        after = max(newer) if newer else scores.get("grpo", scores.get("trained"))
        if base is not None and after is not None and after + 1e-6 < base:
            flags["eval_drop"] = True
            notes = f"best greedy exact match {base} -> {after}"
        elif base is not None and after is not None:
            notes = f"best greedy exact match {after} vs base {base}"
    return {"flags": flags, "notes": notes}


def main() -> None:
    df = load()
    plot(df)
    det = detect(df)
    DET.parent.mkdir(parents=True, exist_ok=True)
    DET.write_text(json.dumps(det, indent=2), encoding="utf-8")
    print(f"figures -> {FIG}")
    print(json.dumps(det, indent=2))


if __name__ == "__main__":
    main()
