#!/usr/bin/env python3
"""Append an eval snapshot into TRAINING_REPORT.md and comparison.json."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "report" / "TRAINING_REPORT.md"
CMP = ROOT / "results" / "baselines" / "comparison.json"
DET = ROOT / "results" / "detectors.json"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--eval", required=True, help="path to run_eval.py json")
    p.add_argument("--tag", required=True, help="baseline | trained | step-N")
    p.add_argument("--notes", default="")
    args = p.parse_args()

    summary = json.loads(Path(args.eval).read_text(encoding="utf-8"))
    CMP.parent.mkdir(parents=True, exist_ok=True)
    cmp = json.loads(CMP.read_text()) if CMP.exists() else {}
    cmp[args.tag] = summary.get("reward_mean")
    CMP.write_text(json.dumps(cmp, indent=2), encoding="utf-8")

    det = json.loads(DET.read_text()) if DET.exists() else {"flags": {}}
    fired = [k for k, v in det.get("flags", {}).items() if v]
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    block = f"""
### Run — {args.tag}

- Date: {now}
- Model: {summary.get("model")}
- Split/level: {summary.get("split")} / {summary.get("level")}
- n: {summary.get("n")}
- task mean: {summary.get("exact_or_task_mean")}
- reward mean: {summary.get("reward_mean")}
- per type: `{json.dumps(summary.get("per_task_type", {}), sort_keys=True)}`
- Detectors fired: {fired or "none"}
- Notes: {args.notes or "—"}
"""
    text = REPORT.read_text(encoding="utf-8") if REPORT.exists() else "# Training report\n"
    marker = "## 4. Run log (append only)"
    if marker in text:
        text = text.replace(marker, marker + "\n" + block, 1)
    else:
        text += "\n" + block
    REPORT.write_text(text, encoding="utf-8")
    print(f"updated {REPORT}")
    print(f"comparison {cmp}")


if __name__ == "__main__":
    main()
