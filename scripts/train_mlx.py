#!/usr/bin/env python3
"""LoRA GRPO on mixed GSM8K train questions, starting from a warmup adapter.

Advantage is the reward minus the group mean. With --tools the reward is
tool_reward. Otherwise it is a matching #### or <answer> line. Truncated
samples are dropped, not scored as zero. A group is skipped unless the best
and worst rewards differ by at least MIN_GAP, and another train question is
drawn. KL is Schulman's k3 against a frozen copy of the reference adapter:
its gradient is zero when the two log-probs match. The policy term uses the
sum of sampled-token log-probs divided by a fixed length. Calculator values
inserted during sampling are not trained on. The test split is never read.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "environments" / "gsm8k_v1"))
sys.path.insert(0, str(ROOT / "scripts"))

from gsm8k_v1.rewards import boxed_match, solution_format  # noqa: E402
from gsm8k_v1.tools import (  # noqa: E402
    TOOL_SYSTEM,
    answer_formula_span,
    calculator_insertion,
    cut_repeated_tail,
    diverge_span,
    expression_f1,
    ordered_prefix_len,
    prefix_end,
    prefix_end_after,
    reference_expressions,
    repeat_start,
    repeated_formula,
    tool_match,
    tool_reward,
    tool_shape,
)
from model_io import chat_prompt, encode, load_policy  # noqa: E402
from train_common import (  # noqa: E402
    attach_lora,
    load_lora_weights,
    save_lora,
    trainable_count,
)

METRICS = ROOT / "results" / "metrics.csv"
ADAPTER = ROOT / "results" / "adapters" / "gsm8k-grpo"
FIELDS = [
    "step",
    "loss",
    "kl",
    "reward_mean",
    "reward_std",
    "pct_all_zero_groups",
    "pct_all_one_groups",
    "pct_zero_std_groups",
    "task_score",
    "format_rate",
    "mean_tokens",
    "updated",
    "reward_gap",
    "hack_rate",
    "tries",
    "n_questions",
]


LENGTH_NORM = 128.0
MIN_GAP = 0.1
MIN_STD = 0.05


def _check_lora_is_live(model, tokenizer) -> None:
    ids = encode(tokenizer, chat_prompt(tokenizer, "What is 1 + 1?"))
    if len(ids) < 12:
        raise SystemExit("tokenizer prompt was too short to check LoRA")
    prompt_ids, comp = ids[:8], ids[8:12]

    def probe(model, prompt_ids, comp):
        return -sequence_logprob(model, prompt_ids, comp)

    _, grads = nn.value_and_grad(model, probe)(model, prompt_ids, comp)
    total = mx.array(0.0)
    for _, grad in tree_flatten(grads):
        total = total + mx.sum(mx.abs(grad))
    mx.eval(total)
    print(f"grad check {float(total):.6f}", flush=True)
    if float(total) < 1e-8:
        raise SystemExit("LoRA received no gradient")


def token_logprobs(model, prompt_ids: list[int], completion_ids: list[int], mask=None):
    """Per-token log-prob of each sampled completion token. Inserted tokens are 0."""
    if not completion_ids:
        return mx.zeros((1,))
    tokens = mx.array(prompt_ids + completion_ids)[None]
    logits = model(tokens)
    start = len(prompt_ids) - 1
    sl = logits[0, start : start + len(completion_ids)]
    logp = nn.log_softmax(sl.astype(mx.float32), axis=-1)
    idx = mx.array(completion_ids)
    chosen = logp[mx.arange(len(completion_ids)), idx]
    if mask is not None:
        chosen = chosen * mx.array(mask, dtype=mx.float32)
    return chosen


def sequence_logprob(model, prompt_ids: list[int], completion_ids: list[int], mask=None):
    """Sum of sampled-token log-probs, divided by a fixed length.

    Dividing by each completion's own length makes a one-token answer a
    larger update than a full solution. A constant denominator does not.
    """
    if not completion_ids:
        return mx.array(0.0)
    return mx.sum(token_logprobs(model, prompt_ids, completion_ids, mask)) / LENGTH_NORM


def kl_penalty(logprob, ref_logprob):
    """Schulman k3. Zero when logprob equals the reference, and nonnegative."""
    ratio = logprob - ref_logprob
    return mx.exp(-ratio) + ratio - 1.0


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="mlx-community/Llama-3.2-1B-Instruct-4bit")
    p.add_argument("--init-adapter", default=str(ROOT / "results" / "adapters" / "gsm8k-sft"))
    p.add_argument("--ref-adapter", default="")
    p.add_argument("--temperature", type=float, default=0.4)
    p.add_argument("--offset", type=int, default=0)
    p.add_argument("--task-file", default=str(ROOT / "results" / "mixed_train.json"))
    p.add_argument("--out", default=str(ADAPTER))
    p.add_argument("--metrics", default=str(METRICS))
    p.add_argument("--steps", type=int, default=20)
    p.add_argument("--group-size", type=int, default=4)
    p.add_argument("--max-tokens", type=int, default=192)
    p.add_argument("--lr", type=float, default=5e-7)
    p.add_argument("--kl-beta", type=float, default=0.01)
    p.add_argument("--lora-rank", type=int, default=8)
    p.add_argument("--lora-layers", type=int, default=8)
    p.add_argument("--tools", action="store_true")
    p.add_argument("--min-winner", type=float, default=0.0)
    p.add_argument("--questions-per-step", type=int, default=4)
    args = p.parse_args()

    import pandas as pd
    import mlx.optimizers as optim
    from mlx_lm import stream_generate
    from mlx_lm.sample_utils import make_sampler

    task_path = Path(args.task_file)
    if not task_path.exists():
        raise SystemExit(f"missing {task_path}; run scripts/select_mixed.py first")
    payload = json.loads(task_path.read_text())
    kept = list(payload.get("kept") or [])
    if not kept:
        raise SystemExit(f"no mixed task ids in {task_path}")

    df = pd.read_parquet(ROOT / "data" / "processed" / "tasks.parquet")
    df = df[df["split"] == "train"].set_index("task_id")
    missing = [task_id for task_id in kept if task_id not in df.index]
    if missing:
        raise SystemExit(f"mixed ids are not in the train split: {missing[:3]}")
    rows = df.loc[kept].reset_index()
    if args.tools:
        keep_idx = [
            i
            for i in range(len(rows))
            if reference_expressions(str(rows.iloc[i]["solution"]), str(rows.iloc[i]["gold"]))
        ]
        if not keep_idx:
            raise SystemExit("no train rows whose formulas end at the official answer")
        rows = rows.iloc[keep_idx].reset_index(drop=True)
        print(f"expression rows {len(rows)}", flush=True)

    init = Path(args.init_adapter)
    ref_path = Path(args.ref_adapter) if args.ref_adapter else init
    print(f"loading reference {ref_path}", flush=True)
    ref_model, _ = load_policy(args.model, str(ref_path))
    ref_model.freeze()
    print(f"loading policy {args.model}", flush=True)
    model, tokenizer = load_policy(args.model)
    attach_lora(model, args.lora_layers, args.lora_rank)
    load_lora_weights(model, init)
    print(f"trainable params {trainable_count(model)} init {init} ref {ref_path}", flush=True)
    print(
        f"grpo questions {len(rows)} lr {args.lr} kl_beta {args.kl_beta} temp {args.temperature}",
        flush=True,
    )

    _check_lora_is_live(model, tokenizer)
    optimizer = optim.Adam(learning_rate=args.lr)
    sampler = make_sampler(temp=args.temperature)

    def force_tokens(prompt_ids, completion_ids, extra: str):
        full = list(prompt_ids) + list(completion_ids)
        decoded = tokenizer.decode(full, skip_special_tokens=False)
        new_ids = encode(tokenizer, decoded + extra)
        if new_ids[: len(full)] != full:
            return None
        return new_ids[len(full) :]

    def sample_one(prompt_ids, greedy: bool = False):
        """Sampled token ids, with calculator values inserted and masked."""
        completion: list[int] = []
        mask: list[float] = []
        spans: list[tuple[int, int]] = []
        text = ""
        remaining = args.max_tokens
        ignored_calc = -1
        while remaining > 0:
            stop = False
            truncated = False
            saw = False
            insert = None
            gen_kwargs = {} if greedy else {"sampler": sampler}
            for response in stream_generate(
                model,
                tokenizer,
                prompt_ids + completion,
                max_tokens=remaining,
                **gen_kwargs,
            ):
                char_at = len(text)
                text += response.text
                if args.tools and (repeated_formula(text) or "</answer>" in text):
                    stop = True
                    break
                if response.finish_reason == "stop":
                    stop = True
                    break
                completion.append(int(response.token))
                mask.append(1.0)
                spans.append((char_at, len(text)))
                saw = True
                remaining -= 1
                if response.finish_reason == "length" or remaining <= 0:
                    truncated = True
                    break
                if not args.tools:
                    continue
                extra = calculator_insertion(text)
                if not extra:
                    continue
                calc_at = text.rfind("calc(")
                if calc_at == ignored_calc:
                    continue
                forced = force_tokens(prompt_ids, completion, extra)
                if not forced:
                    ignored_calc = calc_at
                    continue
                insert = (forced, extra)
                break
            if truncated:
                return None
            if stop or not saw or insert is None:
                break
            forced, extra = insert
            extra_at = len(text)
            completion.extend(forced)
            mask.extend([0.0] * len(forced))
            text += extra
            spans.extend([(extra_at, len(text))] * len(forced))
            if args.tools and (repeated_formula(text) or "</answer>" in text):
                break
            if len(completion) >= args.max_tokens:
                return None
        if not mask or not any(bit == 1.0 for bit in mask):
            return None
        return text, completion, mask, spans

    def loss_fn(model, batch):
        """Policy term plus a per-token KL. The reference log-probs are constants."""
        total = mx.array(0.0)
        used = 0
        for prompt_ids, groups in batch:
            for comp_ids, token_mask, adv, ref_tokens in groups:
                if abs(adv) < 1e-8 or not comp_ids:
                    continue
                pol = token_logprobs(model, prompt_ids, comp_ids, token_mask)
                ref = mx.array(ref_tokens, dtype=mx.float32)
                mask = mx.array(token_mask, dtype=mx.float32)
                penalty = kl_penalty(pol, ref) * mask
                lp = mx.sum(pol) / LENGTH_NORM
                kl = mx.sum(penalty) / LENGTH_NORM
                total = total + (-adv) * lp + args.kl_beta * kl
                used += 1
        if used == 0:
            return mx.array(0.0)
        return total / used

    grad_fn = nn.value_and_grad(model, loss_fn)
    metrics = Path(args.metrics)
    metrics.parent.mkdir(parents=True, exist_ok=True)
    if metrics.exists():
        metrics.unlink()
    t0 = time.time()
    cursor = args.offset % max(len(rows), 1)

    def one_question():
        """One group. Usable only when the samples disagree on the reported score and the best reward has that score."""
        nonlocal cursor
        row = rows.iloc[cursor % len(rows)]
        cursor += 1
        prompt = chat_prompt(
            tokenizer,
            str(row["prompt"]),
            system=TOOL_SYSTEM if args.tools else None,
        )
        prompt_ids = encode(tokenizer, prompt)
        texts, comp_ids_all, masks, span_lists = [], [], [], []

        def _keep(sampled):
            if sampled is None:
                return
            text, comp_ids, token_mask, spans = sampled
            texts.append(text)
            comp_ids_all.append(comp_ids)
            masks.append(token_mask)
            span_lists.append(spans)

        if args.tools:
            # Train on misses only. Already-correct greedy answers are not updated.
            refs = reference_expressions(str(row["solution"]), str(row["gold"]))
            greedy = sample_one(prompt_ids, greedy=True) if refs else None
            _keep(greedy)
            early = False
            if greedy is not None and refs:
                early = ordered_prefix_len(greedy[0], str(row["solution"]), str(row["gold"])) >= 1
            greedy_wrong = greedy is not None and tool_match(cut_repeated_tail(greedy[0]), str(row["gold"])) < 1.0
            # The largest miss is a wrong first formula, so those draws get more samples.
            # A correct first step needs fewer draws: the update can take the next step.
            n_more = 8
            if greedy_wrong:
                for _extra in range(n_more):
                    _keep(sample_one(prompt_ids, greedy=False))
        else:
            for sample_i in range(args.group_size):
                _keep(sample_one(prompt_ids, greedy=(sample_i == 0)))

        def _span_mask(token_mask, text, spans, start, end):
            """Keep sampled tokens whose generated text overlaps [start, end)."""
            if len(spans) != len(token_mask):
                return None
            begin = 0 if start is None else start
            stop = len(text) if end is None else end
            if begin >= stop:
                return None
            new = [0.0] * len(token_mask)
            for index, (left, right) in enumerate(spans):
                if token_mask[index] != 0.0 and right > begin and left < stop:
                    new[index] = 1.0
            if not any(bit == 1.0 for bit in new):
                return None
            return new

        view_texts, view_ids, pos_masks, neg_masks = [], [], [], []
        rewards, bits, prefixes = [], [], []
        if args.tools:
            gold = str(row["gold"])
            solution = str(row["solution"])
            question = str(row["prompt"])
            greedy_text = cut_repeated_tail(texts[0]) if texts else ""
            greedy_prefix = ordered_prefix_len(greedy_text, solution, gold) if texts else 0

            def _add(text, ids, token_mask, reward_text, pos, neg, reward):
                view_texts.append(reward_text)
                view_ids.append(ids)
                pos_masks.append(pos if pos is not None else [0.0] * len(token_mask))
                neg_masks.append(neg if neg is not None else [0.0] * len(token_mask))
                rewards.append(reward)
                bits.append(tool_match(reward_text, gold))
                prefixes.append(ordered_prefix_len(reward_text, solution, gold))

            for text, ids, token_mask, spans in zip(texts, comp_ids_all, masks, span_lists):
                scored = cut_repeated_tail(text)
                answer_span = answer_formula_span(text, solution, gold)
                if answer_span is not None:
                    answer_at = answer_span[1]
                    cut = text[:answer_at].rstrip()
                    # Reinforce the answer formula itself, and push down the formulas after it.
                    # Do not reinforce earlier wrong steps that happened to precede it.
                    prefix_mask = _span_mask(token_mask, text, spans, answer_span[0], answer_at)
                    tail_mask = _span_mask(token_mask, text, spans, answer_at, None)
                    if (
                        prefix_mask is not None
                        and tool_match(cut, gold) == 1.0
                        and expression_f1(cut, solution, question) > 0.0
                    ):
                        _add(
                            text, ids, token_mask, cut,
                            prefix_mask,
                            [0.0] * len(token_mask),
                            tool_reward(cut, gold, solution, question),
                        )
                        if tail_mask is not None:
                            _add(
                                text, ids, token_mask, text,
                                [0.0] * len(token_mask),
                                tail_mask,
                                0.0,
                            )
                        continue
                end = prefix_end(scored, solution, gold)
                diverge = diverge_span(scored, solution, gold)
                repeated = repeat_start(text)
                # Reinforce only the steps past the greedy prefix, so a new last
                # formula does not rewrite a first step the greedy answer already had.
                shared = prefix_end_after(text, solution, gold, greedy_prefix)
                if greedy_prefix and shared is None:
                    start_at = None
                else:
                    start_at = shared or 0
                if start_at is None:
                    pos = [0.0] * len(token_mask)
                elif tool_match(scored, gold) >= 1.0 and ordered_prefix_len(scored, solution, gold) >= greedy_prefix:
                    stop_at = repeated if repeated is not None else None
                    pos = _span_mask(token_mask, text, spans, start_at, stop_at)
                elif end and tool_match(scored, gold) < 1.0:
                    pos = _span_mask(token_mask, text, spans, start_at, end)
                elif repeated is not None:
                    pos = _span_mask(token_mask, text, spans, start_at, repeated)
                else:
                    pos = _span_mask(token_mask, text, spans, start_at, None)
                neg = _span_mask(token_mask, text, spans, diverge[0], diverge[1]) if diverge else None
                _add(text, ids, token_mask, scored, pos, neg, tool_reward(scored, gold, solution, question))
        else:
            view_texts = list(texts)
            view_ids = list(comp_ids_all)
            pos_masks = list(masks)
            neg_masks = list(masks)
            rewards = [boxed_match(text, str(row["gold"])) for text in texts]
            bits = list(rewards)
            prefixes = [0 for _text in texts]
        fmts = [solution_format(text) for text in view_texts]
        gap = (max(rewards) - min(rewards)) if len(rewards) >= 2 else 0.0
        if args.tools and rewards:
            greedy_text = cut_repeated_tail(texts[0]) if texts else ""
            greedy_match = tool_match(greedy_text, str(row["gold"])) if texts else 1.0
            greedy_prefix = ordered_prefix_len(greedy_text, str(row["solution"]), str(row["gold"])) if texts else 0

            def _rank(index):
                finished = 0
                if bits[index] > 0.0 and expression_f1(
                    view_texts[index], str(row["solution"]), str(row["prompt"])
                ) > 0.0:
                    finished = 1
                return (finished, prefixes[index], rewards[index])

            winner_i = max(range(len(rewards)), key=_rank)
            finished_win = _rank(winner_i)[0] == 1
            mean_r = sum(rewards) / len(rewards)
            std_r = (sum((reward - mean_r) ** 2 for reward in rewards) / len(rewards)) ** 0.5
            same_path = prefixes[winner_i] >= greedy_prefix
            # A finished sample must extend the greedy path, clear the reward gap,
            # and not be a near-tie. Tiny standard deviations are skipped, not divided out.
            winner_ok = (
                greedy_match < 1.0
                and finished_win
                and same_path
                and gap + 1e-9 >= MIN_GAP
                and std_r + 1e-12 >= MIN_STD
            )
            win_kind = "finished" if winner_ok else "none"
            split = len(rewards) >= 2 and min(bits) < 1.0 and max(bits) > 0.0
        else:
            split = len(rewards) >= 2 and min(bits) < 1.0 and max(bits) > 0.0
            winner_i = rewards.index(max(rewards)) if rewards else 0
            winner_ok = bool(rewards) and bits[winner_i] > 0.0 and gap + 1e-9 >= MIN_GAP
            win_kind = "finished" if winner_ok else "none"
        if args.min_winner > 0 and rewards:
            winner_ok = winner_ok and max(rewards) + 1e-9 >= args.min_winner
        usable = split and winner_ok
        return {
            "row": row,
            "prompt_ids": prompt_ids,
            "texts": view_texts,
            "comp_ids": view_ids,
            "masks": pos_masks,
            "neg_masks": neg_masks,
            "rewards": rewards,
            "bits": bits,
            "fmts": fmts,
            "gap": gap,
            "usable": usable,
            "win_kind": win_kind,
        }

    detail_path = metrics.with_name(metrics.stem + "_completions.jsonl")
    if detail_path.exists():
        detail_path.unlink()
    with metrics.open("w", encoding="utf-8", newline="") as handle, detail_path.open(
        "w", encoding="utf-8"
    ) as detail:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        for step in range(args.steps):
            picked = []
            tries = 0
            while len(picked) < args.questions_per_step and tries < args.questions_per_step * 15:
                tries += 1
                pack = one_question()
                if pack["usable"]:
                    picked.append(pack)
            rewards = [reward for pack in picked for reward in pack["rewards"]]
            texts = [text for pack in picked for text in pack["texts"]]
            fmts = [fmt for pack in picked for fmt in pack["fmts"]]
            gap = sum(pack["gap"] for pack in picked) / len(picked) if picked else 0.0
            if rewards:
                mean_r = sum(rewards) / len(rewards)
                var = sum((reward - mean_r) ** 2 for reward in rewards) / len(rewards)
                std_r = var ** 0.5
            else:
                mean_r, std_r = 0.0, 0.0
            hack_rate = 0.0
            if args.tools and texts:
                hacks = 0
                for pack in picked:
                    for text in pack["texts"]:
                        matched = tool_match(text, str(pack["row"]["gold"]))
                        overlap = expression_f1(text, str(pack["row"]["solution"]), str(pack["row"]["prompt"]))
                        if matched == 1.0 and overlap == 0.0:
                            hacks += 1
                hack_rate = hacks / len(texts)
            kl_value = 0.0
            loss_value = 0.0
            updated = 0
            batch = []
            kl_terms = []
            if picked:
                for pack in picked:
                    group_mean = sum(pack["rewards"]) / len(pack["rewards"])
                    groups = []
                    for comp_ids, pos_mask, neg_mask, reward in zip(
                        pack["comp_ids"], pack["masks"], pack["neg_masks"], pack["rewards"]
                    ):
                        adv = reward - group_mean
                        token_mask = pos_mask if adv > 0 else neg_mask
                        if abs(adv) < 1e-8 or not any(token_mask):
                            continue
                        ref_vec = token_logprobs(ref_model, pack["prompt_ids"], comp_ids, token_mask)
                        mx.eval(ref_vec)
                        ref_tokens = [float(value) for value in ref_vec]
                        groups.append((comp_ids, token_mask, adv, ref_tokens))
                        pol_vec = token_logprobs(model, pack["prompt_ids"], comp_ids, token_mask)
                        mx.eval(pol_vec)
                        pen = kl_penalty(pol_vec, mx.array(ref_tokens)) * mx.array(token_mask)
                        mx.eval(pen)
                        kl_terms.append(float(mx.sum(pen) / LENGTH_NORM))
                    if groups:
                        batch.append((pack["prompt_ids"], groups))
                if kl_terms:
                    kl_value = sum(kl_terms) / len(kl_terms)
                if batch:
                    loss, grads = grad_fn(model, batch)
                    optimizer.update(model, grads)
                    mx.eval(model.parameters(), loss)
                    loss_value = float(loss)
                    if loss_value != loss_value:
                        raise SystemExit(f"grpo loss became nan at step {step}")
                    updated = 1
            lengths = [len(text.split()) for text in texts] or [0]
            rec = {
                "step": step,
                "loss": round(loss_value, 4),
                "kl": round(kl_value, 4),
                "reward_mean": round(mean_r, 4) if rewards else 0.0,
                "reward_std": round(std_r, 4),
                "pct_all_zero_groups": float(bool(rewards) and all(reward == 0 for reward in rewards)),
                "pct_all_one_groups": float(bool(rewards) and all(reward >= 0.99 for reward in rewards)),
                "pct_zero_std_groups": float(not picked),
                "task_score": round(mean_r, 4) if rewards else 0.0,
                "format_rate": round(sum(fmts) / len(fmts), 4) if fmts else 0.0,
                "mean_tokens": round(sum(lengths) / len(lengths), 2),
                "updated": updated,
                "reward_gap": round(gap, 4),
                "hack_rate": round(hack_rate, 4),
                "tries": tries,
                "n_questions": len(picked),
            }
            writer.writerow(rec)
            handle.flush()
            if args.tools:
                for pack in picked:
                    gold = str(pack["row"]["gold"])
                    solution = str(pack["row"]["solution"])
                    question = str(pack["row"]["prompt"])
                    best = max(pack["rewards"]) if pack["rewards"] else None
                    for text, reward in zip(pack["texts"], pack["rewards"]):
                        matched = tool_match(text, gold)
                        overlap = expression_f1(text, solution, question)
                        shape = tool_shape(text, solution, question)
                        detail.write(
                            json.dumps(
                                {
                                    "step": step,
                                    "task_id": str(pack["row"]["task_id"]),
                                    "match": matched,
                                    "overlap": round(overlap, 4),
                                    "shape": round(shape, 4),
                                    "reward": round(reward, 4),
                                    "winner": reward == best and matched == 1.0 and overlap > 0.0,
                                }
                            )
                            + "\n"
                        )
                detail.flush()
            print(
                f"step {step:03d} loss={rec['loss']} kl={rec['kl']} reward={rec['reward_mean']} "
                f"gap={rec['reward_gap']} hack={rec['hack_rate']} questions={len(picked)} "
                f"updated={updated} tries={tries} "
                f"wins={','.join(pack['win_kind'] for pack in picked) or '-'}",
                flush=True,
            )

    out = Path(args.out)
    save_lora(model, out, args.lora_layers, args.lora_rank)
    print(f"saved {out} in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
