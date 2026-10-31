"""GSM8K rewards. Gold is the number after ####. Score the parsed final answer."""

from __future__ import annotations

import re

ANSWER_RE = re.compile(r"<answer>(.*?)</answer>", re.IGNORECASE | re.DOTALL)
BOX_RE = re.compile(r"\\boxed\{([^{}]+)\}")
HASH_RE = re.compile(r"####\s*([-+]?\d[\d,]*(?:\.\d+)?)")
NUM_RE = re.compile(r"[-+]?\d[\d,]*(?:\.\d+)?")


def gold_from_solution(solution: str) -> str:
    m = HASH_RE.search(solution or "")
    raw = m.group(1) if m else (solution or "")
    return normalize_number(raw)


def normalize_number(text: str) -> str:
    t = (text or "").strip().replace(",", "").replace("$", "")
    m = NUM_RE.search(t)
    if not m:
        return ""
    n = m.group(0)
    if "." in n:
        n = n.rstrip("0").rstrip(".")
    if n in {"-0", "+0"}:
        n = "0"
    return n.lstrip("+")


def extract_answer(text: str) -> str:
    if not text:
        return ""
    tagged = ANSWER_RE.search(text)
    if tagged:
        return tagged.group(1).strip()
    boxes = BOX_RE.findall(text)
    if boxes:
        return boxes[-1].strip()
    hashes = HASH_RE.findall(text)
    if hashes:
        return hashes[-1]
    nums = NUM_RE.findall(text)
    return nums[-1] if nums else ""


def format_box(completion: str) -> float:
    return 1.0 if ANSWER_RE.search(completion or "") else 0.0


def solution_format(completion: str) -> float:
    """1 when the completion has an <answer> tag, a \\boxed{}, or a #### line."""
    text = completion or ""
    return 1.0 if (ANSWER_RE.search(text) or BOX_RE.search(text) or HASH_RE.search(text)) else 0.0


def exact_match(completion: str, gold: str) -> float:
    """1 only when an <answer> tag, \\boxed{}, or #### line matches.

    A trailing number is not an answer. "5:00" must not become 0.
    """
    return boxed_match(completion, gold)


def boxed_match(completion: str, gold: str) -> float:
    """Credit only a matching <answer> tag or #### line. A bare number scores 0."""
    text = completion or ""
    tagged = ANSWER_RE.search(text)
    boxes = BOX_RE.findall(text)
    hashes = HASH_RE.findall(text)
    if tagged:
        got = tagged.group(1)
    elif boxes:
        got = boxes[-1]
    elif hashes:
        got = hashes[-1]
    else:
        return 0.0
    return 1.0 if normalize_number(got) == normalize_number(str(gold)) else 0.0


def score_task(completion: str, gold: str, gold_type: str = "exact") -> float:
    if gold_type != "exact":
        raise ValueError(f"gsm8k only supports exact, got {gold_type}")
    return exact_match(completion, gold)


def combined(completion: str, gold: str, gold_type: str = "exact", format_weight: float = 0.1) -> float:
    task = score_task(completion, gold, gold_type)
    fmt = format_box(completion)
    return (task + format_weight * fmt) / (1.0 + format_weight)
