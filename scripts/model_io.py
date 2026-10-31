"""Shared MLX load + chat prompt for benchmark and GRPO."""

from __future__ import annotations

from pathlib import Path

SYSTEM = (
    "Solve the grade-school math problem. "
    "Put the final number only in <answer>...</answer>."
)


def chat_prompt(tokenizer, question: str, system: str | None = None) -> str:
    user = (system or SYSTEM) + "\n\n" + question
    if hasattr(tokenizer, "apply_chat_template"):
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": user}],
            tokenize=False,
            add_generation_prompt=True,
        )
    return user


def few_shot_prompt(tokenizer, question: str, exemplars: list[tuple[str, str]], system: str | None = None) -> str:
    """Chat prompt with train-only worked examples before the question."""
    messages = []
    for sample_question, sample_answer in exemplars:
        messages.append({"role": "user", "content": (system or SYSTEM) + "\n\n" + sample_question})
        messages.append({"role": "assistant", "content": sample_answer})
    messages.append({"role": "user", "content": (system or SYSTEM) + "\n\n" + question})
    if hasattr(tokenizer, "apply_chat_template"):
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return "\n\n".join(item["content"] for item in messages)


def encode(tokenizer, text: str, add_special_tokens: bool = False) -> list[int]:
    """Token ids for text. Chat prompts already contain the start token.

    The default leaves that token alone. Encoding again with special tokens
    on inserts a second start token, and the log-probability is then for a
    different string than the one that was sampled.
    """
    ids = tokenizer.encode(text, add_special_tokens=add_special_tokens)
    if isinstance(ids, list) and ids and isinstance(ids[0], list):
        ids = ids[0]
    return list(ids)


def supervised_ids(
    tokenizer, question: str, solution: str, system: str | None = None
) -> tuple[list[int], int] | None:
    """Token ids for question + train solution, and the prompt length.

    The solution is the assistant turn, including the closing end-of-turn token.
    Returns None if the prompt is not a token prefix of the full string.
    """
    prompt = chat_prompt(tokenizer, question, system=system)
    solution = (solution or "").strip()
    if not solution:
        return None
    user = (system or SYSTEM) + "\n\n" + question
    if hasattr(tokenizer, "apply_chat_template"):
        full = tokenizer.apply_chat_template(
            [
                {"role": "user", "content": user},
                {"role": "assistant", "content": solution},
            ],
            tokenize=False,
            add_generation_prompt=False,
        )
    else:
        full = prompt + solution
    prompt_ids = encode(tokenizer, prompt)
    full_ids = encode(tokenizer, full)
    if full_ids[: len(prompt_ids)] != prompt_ids:
        return None
    if len(full_ids) <= len(prompt_ids):
        return None
    return full_ids, len(prompt_ids)


def load_policy(model_id: str, adapter: str | None = None):
    from mlx_lm import load

    if adapter and Path(adapter).exists():
        return load(model_id, adapter_path=adapter)
    return load(model_id)
