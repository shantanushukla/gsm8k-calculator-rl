"""GSM8K taskset. Train on split=train. Compare base vs trained on split=test."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from gsm8k_v1.rewards import combined

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_TASKS = ROOT / "data" / "processed" / "tasks.parquet"

SYSTEM = (
    "Solve the grade-school math problem. "
    "Put the final number only in <answer>...</answer>."
)


def load_task_frame(split: str, tasks_path: Path = DEFAULT_TASKS) -> pd.DataFrame:
    if not tasks_path.exists():
        return pd.DataFrame()
    df = pd.read_parquet(tasks_path)
    if split and "split" in df.columns:
        df = df[df["split"] == split]
    return df.reset_index(drop=True)


class Gsm8kTaskset:
    def __init__(self, split: str = "train", tasks_path: str | None = None):
        self.split = split
        self.tasks_path = Path(tasks_path) if tasks_path else DEFAULT_TASKS

    def load_frames(self) -> pd.DataFrame:
        return load_task_frame(self.split, self.tasks_path)

    def score(self, completion: str, gold: Any, gold_type: str = "exact") -> float:
        return combined(completion, str(gold), gold_type)


try:
    import verifiers.v1 as vf

    class Gsm8kData(vf.TaskData):
        gold: str
        gold_type: str = "exact"

    class Gsm8kTask(vf.Task[Gsm8kData]):
        @vf.reward
        async def task_reward(self, trace: vf.Trace) -> float:
            text = getattr(trace, "last_reply", "") or ""
            return combined(text, self.data.gold, self.data.gold_type)

        @vf.metric
        async def has_answer_tag(self, trace: vf.Trace) -> float:
            from gsm8k_v1.rewards import format_box

            return format_box(getattr(trace, "last_reply", "") or "")

    class Gsm8kConfig(vf.TasksetConfig):
        split: str = "test"
        tasks_path: str = str(DEFAULT_TASKS)

    class Gsm8kVfTaskset(vf.Taskset[Gsm8kTask, Gsm8kConfig]):
        def load(self) -> list[Gsm8kTask]:
            df = load_task_frame(self.config.split, Path(self.config.tasks_path))
            tasks: list[Gsm8kTask] = []
            for i, row in df.iterrows():
                data = Gsm8kData(
                    idx=int(i),
                    prompt=SYSTEM + "\n\n" + str(row["prompt"]),
                    gold=str(row["gold"]),
                    gold_type="exact",
                )
                tasks.append(Gsm8kTask(data, self.config.task))
            return tasks

    Gsm8kTaskset = Gsm8kVfTaskset  # type: ignore[misc, assignment]

except ImportError:
    pass
