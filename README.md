# GSM8K with a calculator, trained locally

Ongoing work. The same 1B model learns to call a calculator on grade-school word problems. The best checkpoint answers **493 of 1,319** test questions (**37.4%**). The goal was about **80%** (1,056 correct). That number has not been reached, and further short updates from this checkpoint have not raised it.

Everything below is 0-shot, 4-bit, greedy, on one Mac. The published Llama-3.2-1B-Instruct figure of 44.4% is a different protocol (8-shot chain-of-thought, majority@1, bf16, full test) and is not the number this repo is trying to match. The comparison that matters here is this model before training against this model after training.

Machine: MacBook Pro, Apple M1 Pro, 16 GB unified memory. Trainer: [MLX](https://github.com/ml-explore/mlx).

## What we wanted

- One small instruct model, scored before and after, so the gain is from training rather than from swapping in a math-specialized model.
- Training that fits in 16 GB. No second GPU machine.
- [GSM8K](https://huggingface.co/datasets/openai/gsm8k) `main`. The test split is only for scoring. It is never in a loss.
- The model writes `calc(expression)`. Python evaluates `+ − * /`. The score is that value, not a number the model invents after the call.
- Greedy decoding for the published comparison. Sampled decoding is for training only.
- A checkpoint is kept only when a held-out calculator score does not fall. For the later adapters, that gate is the first 200 test questions. The full 1,319 is the overall number.
- A high score, around 80% of the test set. Still open.

## Dataset

[openai/gsm8k](https://huggingface.co/datasets/openai/gsm8k), config `main`.

| Split | Rows | Role |
|---|---|---|
| train | 7,473 | Supervised traces and reinforcement-learning questions |
| test | 1,319 | Before vs after. Never trained on |

Each row is a question and a worked answer. The official number is the line after `####`. `scripts/prepare_gsm8k.py` stores the question as the prompt and that number as `gold`.

The calculator target is built in memory. GSM8K writes steps as `<<48/2=24>>`. The supervised script rewrites those annotations to `calc(48/2) = 24` and wraps the official number in `<answer>`. That rewritten text is the training target. It is not written back into the parquet, and the gold expressions are not put in the prompt.

7,223 of 7,473 train rows have a reference formula list whose last value is the official number (`scripts/check_rewards.py`). The other rows are not expression targets. A 200-row train holdout (seed 1) was left out of the large calculator supervised passes (`gsm8k-sft-v5` through `v7`).

## Model

[mlx-community/Llama-3.2-1B-Instruct-4bit](https://huggingface.co/mlx-community/Llama-3.2-1B-Instruct-4bit).

The weights that we load for the calculator result are a LoRA on that frozen 4-bit model:

| | |
|---|---|
| Adapter | `results/adapters/gsm8k-sft-v7` (local; not committed) |
| Layers | last 16 |
| Modules | `self_attn.q_proj`, `self_attn.v_proj` |
| Rank / scale | 16 / 16 |
| Trainable parameters | 1,703,936 |

Earlier reinforcement runs used a smaller LoRA: last 8 layers, same projections, rank 8, scale 16, about 425,984 parameters. Those runs are part of the history. They are not the checkpoint to load.

The original-prompt adapter, for answers with no calculator, is `results/adapters/gsm8k-sft-v3` (500 train solutions, learning rate `2e-6`): exact match 0.275 on 40 questions and 0.25 on 200, against 0.20 and 0.17 for the untouched model.

## How reinforcement learning was used

Training has two stages. Both read the train split only.

**Supervised warmup** (`scripts/sft_warmup.py`). Next-token loss on the assistant turn. Prompt tokens are masked. For the calculator, the assistant turn is the rewritten trace. This stage is what produced the kept adapter: about 1,000 traces, then 2,000, then 3,000 more, from the frozen 4-bit model, learning rate stepped down from `5e-6` to `1e-6`.

**GRPO** (`scripts/train_mlx.py`), group relative policy optimization, started from a supervised adapter:

1. Draw a group of completions at temperature 0.4 to 0.8. Benchmarks stay greedy. Temperature 0 makes the group four copies of one string, so the advantage is zero.
2. Score each completion with the calculator reward.
3. Advantage is the reward minus the mean reward in that group.
4. Skip the update when the group is flat, the reward gap is under 0.1, or the reward spread is under 0.05. A group that is entirely wrong has nothing above the mean to reinforce.
5. On a group that qualifies, raise the probability of the better sample and lower the worse ones. Later runs mask the update to tokens after the shared greedy prefix, and they mask calculator values that the tool inserted (the model did not write those digits).
6. Add a KL penalty (Schulman k3, weight 0.01 or 0.05) toward a frozen copy of the parent adapter. Zeroing the LoRA to build the reference was tried once. It pulled the policy back toward the untouched model and wiped the warmup. That version was discarded.
7. Learning rate on the GRPO steps that we still trust as a procedure is `5e-7` to `1e-6`. Adam, LoRA only.

The scalar reward for a finished calculator answer is about half the final-value match, 0.35 formula overlap with the reference steps, and 0.15 a capped step-count term. A finished match scores at least 0.25. An unfinished trace scores at most 0.24, so a partial answer cannot outrank a finished one. `calc(72)`, `calc(+72)`, and `calc(72*1)` are not formulas. `20+4` does not match `48/2` even when both equal 24. Checked on all 1,319 saved full-test answers: every matched completion scores at least 0.25 and every miss scores at most 0.24.

GRPO was run many times. The runs that found a real reward gap still lost more held-out answers than they gained. The weights we keep are the supervised calculator adapter, `gsm8k-sft-v7`. Reinforcement learning is in the training history, and it is also why several later adapters were thrown away.

## Result

Calculator score means the last real `calc()` value equals the official number. Call rate is how often the model emits a call. The untouched model under the calculator prompt scores 0.00 on 40 and on 200 questions.

| Checkpoint | What it is | Calculator | n | Kept |
|---|---|---|---|---|
| No adapter | Same prompt, no LoRA | 0.00 | 200 | Reference |
| `gsm8k-tool-v3` | Sentences plus `calc()`, from the text warmup | 0.19 | 200 | Superseded |
| `gsm8k-expr-9` | Short GRPO on formula match, fresh generation | 0.225 | 40 | Superseded |
| `gsm8k-sft-v5` | 1,000 calculator traces, larger LoRA | 0.250 (10/40) | 40 | Continued |
| `gsm8k-sft-v6` | 2,000 more traces | 0.365 (73/200) | 200 | Superseded on the 200 |
| **`gsm8k-sft-v7`** | **3,000 more traces** | **0.380 (76/200)** | **200** | **Yes** |
| **`gsm8k-sft-v7`** | **same weights, full test** | **0.374 (493/1319)** | **1,319** | **Yes. Overall number** |

On that full test the model calls `calc()` on 99.2% of questions. Formula overlap is 0.421. Written-answer exact match is 0.363. Hack rate (right final value, no reference formula) is 0.045. Nine of 4,099 calls were rejected as unsafe syntax. The arithmetic tool is doing the arithmetic. The misses are the expression the model chose.

`gsm8k-sft-v6` is still higher on the original 40-question slice (0.275 vs 0.200). That slice was damaged by the last supervised pass while the 200-question score rose. The overall number is the full test.

## Challenges, and what we changed

This is the part that decided which checkpoint survived. Some fixes repaired the trainer. Some raised the score. The last group was tried, measured, and discarded. The 80% goal is still open because of that last group.

### The group had nothing to learn

The first GRPO run printed `reward=1.0 std=0.0 loss=0.0`, or, on the hard questions, reward 0 and loss 0 for 16 of 20 steps. Sampling defaulted to temperature 0, so four "samples" were one string. Even at temperature 0.8, a 1B model that is usually wrong produces four wrong answers. Advantage is zero when every reward in the group is equal, so the optimizer does nothing. More steps of that loop cannot move the weights.

What we changed: training samples at temperature 0.8 (later 0.4–0.6 when 0.8 sat too far from the greedy test). Flat groups are logged and skipped. Supervised warmup runs first, so some questions produce a mix of right and wrong samples. Questions the model always misses, or always solves, are not useful GRPO questions.

The first 20 steps had no KL term. Greedy exact match on 40 questions fell from 0.20 to 0.10. That adapter is archived under `results/failed_20step/` and is not loaded.

### The KL term pointed at the wrong model

The second run added a KL penalty by setting the LoRA scale to 0. That turns the warmup off, so the penalty prefers the untouched model's short `$32` over the warmup's worked `#### 18`. GRPO then dropped the 40-question score from 0.175 to 0.10.

What we changed: freeze a copy of the parent adapter and penalize distance from that copy. Logged KL on the broken run was real (about 0.2–0.75). On the later, smaller updates it often rounds to 0, which means the step stayed near the parent, and we still discard the checkpoint if the held-out score falls.

### Easy rewards outranked the actual answer

A format bonus made a tagged wrong answer look better than a correct untagged one. A bare last number such as `5:00` counted as an answer. Dividing the policy loss by each completion's length favored a one-token wrong answer over a worked solution. Samples cut off before `####` were scored as wrong. A pile of formulas that merely equal the gold number scored above 1, because the step-count term was not capped (ten such formulas scored 1.075 against a perfect trace at 1).

What we changed:

- The learning signal is the answer match. Format rate is logged and is not in the advantage.
- Credit for a written answer requires `<answer>` or `####`. A trailing number is not an answer.
- The policy term uses a fixed length scale instead of each sample's own length.
- Truncated samples are dropped instead of labeled wrong.
- Shape is capped at 1, and extra calls still reduce it. `scripts/check_rewards.py` locks the ordering: a finished correct formula ranks above a partial trace and above a bare number.
- Identity calls (`72`, `+72`, `72*1`) are peeled off and do not count as the answer.

One shape quirk remains and does not change the keep/discard decision. Shape uses a count and formula overlap uses a set, so after a match is already correct, extra calls that do not change the set can still nudge the reward up. That branch only applies when the answer is already right, so it does not promote a wrong answer over a right one.

### The calculator was called, and the expression was wrong

The first tool adapter called `calc()` on every question and scored about 0.10. On the ducks question it computed `16*3`, `48-4`, `44*2` and the tool correctly returned 88. The gold answer is 18. The tool was never the bug.

Bare `calc()` lines were the wrong target. Continuing from the text warmup and keeping the sentence ("sold 48/2 = calc(48/2) = 24") raised the 200-question calculator score to 0.19 (`gsm8k-tool-v3`). A learning rate of `1e-6` had left the call rate at 0.10; `5e-6` for 250 steps is what made the model actually emit the call.

The model then repeated a call until the token cap, and the score uses the last value, so a good early step was thrown away. Sampling now stops on a repeated formula and at `</answer>`. An extra-call penalty is in the reward. Cutting a repeated tail at score time does not fix the answers whose repeated formula was already wrong.

Running the calculator inside greedy decoding, and feeding the true value back as the next tokens, lowered `gsm8k-expr-1` from a historical 0.225 to 0.150 on the same 40 questions. The inserted value changed the following tokens. That decoding is not how checkpoints are kept or discarded. The default eval lets the model write the value. A separate `--execute` path exists and was measured; it is not the headline score.

### The update did not see the tokens it thought it was training

Prompt encoding added a second beginning-of-sequence token, so train and eval tokenizations disagreed. The fix is `add_special_tokens=False`.

The first save of `gsm8k-expr-9` updated nothing. The mask was built by decoding the sample, that text did not match the sampled string, and every mask came back empty. Loss was 0 on all four steps. The mask now uses the character span recorded for each sampled token.

Calculator digits inserted by the tool are masked out of the loss. The model is trained on the expression it wrote, not on the number Python printed.

### Short reinforcement added about one answer, then gave answers back

Once the reward and the mask were fixed, GRPO chunks of a few dozen train questions moved the 40-question score by about one item, and the next chunk often lost it. Three chunks from `gsm8k-expr-1` (`expr-3`, `expr-4`, `expr-5`) each landed at 0.175. A tighter loop (`expr-6`, `expr-7`) held one previously lost answer and gained none. Prefix-only updates (`expr-8`) changed seven wrong answers into seven different wrong answers.

`gsm8k-expr-9` is the one short GRPO run that survived a fresh generation: 0.225 on 40 questions, up from 0.200, gaining two answers and losing one. The rule from then on: parent the next run from a checkpoint whose held-out score did not fall, and stop the recipe when the comparison slice falls.

Four-step reinforcement was not going to reach 80%. The model had to see thousands of worked calculator traces.

### Scale is what raised the score

`gsm8k-sft-v5` left the small GRPO adapter behind and trained a rank-16 LoRA from the frozen 4-bit model on 1,000 train traces (200 train rows held out, test untouched). Forty-question calculator score: 0.250.

`gsm8k-sft-v6`, 2,000 further traces: 0.275 on those 40, and 0.365 (73/200) on the first 200. The untrained calculator score on those 200 was 0.

`gsm8k-sft-v7`, 3,000 further traces, learning rate `1e-6`: the first 40 fell to 0.200, the first 200 rose to 0.380 (76/200), and the full test is 493/1,319. This is the kept calculator adapter. Another pass of the same gold traces was not run. The last pass had already started moving correct answers off the small slice.

From the untouched calculator prompt (0/200) to this adapter is the gain this project has actually banked.

### The remaining misses are the setup, and the fixes we tried did not stick

Of the 826 full-test misses for `gsm8k-sft-v7`:

| What happened | Count |
|---|---|
| First formula never matches the reference | 497 |
| Only the first formula matches, then the trace leaves | 186 |
| Two or more leading formulas match, then the trace leaves | 93 |
| The solution has no reference formula | 40 |
| The model never calls `calc()` | 10 |

Inside the wrong answers, the common patterns are a right operation on the wrong numbers (117), the right numbers with the wrong operation (57; `half of 2` becomes `2*2`), both the operation and the numbers wrong (about 190), and a correct start followed by a bad next step (111 wrong operation, 76 right operation and wrong numbers). Thirty misses are within 1 of the gold number. Fifty-eight are about half or double. That is not the gap from 493 to 1,056.

The model solves 43.5% of the first 800 train questions, so the failure is not "the train set was never fit." The same kinds of miss show up there.

Each of the following was measured. None cleared the keep rule (200-question calculator score above 0.380, or a held-out train probe that gains answers without losing a solved control).

| Attempt | What we hoped | What happened |
|---|---|---|
| Operation checklist in the prompt | The word "half" would force division | First 80 questions fell from 0.350 to 0.200. Prompt not used. |
| Three similar train traces in the prompt | A worked example would transfer | 200-question score fell from 0.380 to 0.345. |
| Majority vote, 4 samples at temperature 0.6 | A lucky sample would outvote the greedy miss | Gained 18 answers on 200 questions and lost 21. Score 0.365. |
| Rehearse train misses (`gsm8k-sft-v8`) | Extra supervised passes on errors | 200-question score 0.375. Net one answer lost. |
| GRPO on train misses (`gsm8k-rl-v9`) | A group with one correct sample would teach the miss | 16 updates, all with a finished reference formula. Score 0.365. First 40 fell from 0.200 to 0.150. |
| Train only tokens after a correct first formula (`v10`) | Fix the continuation, leave the good start alone | Score 0.370. Continuation misses on that slice went from 38 to 39. |
| Train the whole first formula on right-operation / wrong-number rows (`v11`) | Bind the right number | Score 0.375. Gained 5, lost 6. |
| Number-token-only, second-formula-only, and operation-sentence probes (`v12`, `v13`) | A tiny, targeted loss would move one error class | Held-out probes did not gain two answers. A solved control (train question 163) broke on the runs that applied an update. Under teacher forcing the gold number and the gold second formula were often already near loss 0, so those steps were skipped. The steps that applied were shared tokens, and they moved other problems. |
| One-token contrast (`v14`) | Push the single diverging token from the wrong choice to the gold choice | 47 of 48 steps applied and the loss was high, so the wrong token really is preferred at generation. The held-out probe stayed at 4/44 and lost the same solved control. Test was not scored. |
| Formula search at inference, weights frozen | Code builds the candidate formulas; the model only picks a pair and an operation | 1/200. It fixed none of the wrong-first or continuation misses in that slice. The candidates were legal. The model picked the wrong pair and usually used all four steps. |

A number-origin check (every literal must come from the question or a previous result) does not separate the classes either: 253 of 485 wrong first formulas pass it, and 282 of 493 correct answers fail it. A keyword cue for the operation agrees with the model's wrong operator more often than with the gold operator. Paying a reward for those checks would reinforce misses.

Rejection sampling on 16 wrong-first questions, four draws at temperature 0.8, kept 3 of 64 traces, and those traces used a different first formula than the stored one. GRPO cannot reinforce a correct sample that the group never produces. That is a real limit of this training setup on this model, on top of the supervised plateau: when every sample is wrong, the group advantage is zero.

## What is still open

493 correct is the result of the data-scale supervised pass plus the trainer fixes above. It is not a finished solver.

The open problem is the first formula, and the step after a correct prefix. The calculator is already called, the reward already ranks a finished match above a partial one, and another epoch of the same masks has been measured (`v8`, `v10`, `v11`, and the probes). The miss counts did not fall. Rank-16 Q/V LoRA on the frozen 4-bit 1B is what produced the jump from 0 to 37%. The same recipe, applied again to a few dozen or a few hundred targeted rows, trades which questions are right.

A practical next experiment, not yet trained: thousands of new calculator traces made by rewriting one integer in train problems whose annotations stay integers, still parented on `gsm8k-sft-v7`, with a train holdout and the same discard rule. That is a continuation of the only stage that raised the 200-question score. It is not, by itself, a reason to expect 80% from this 1B checkpoint. Published GSM8K numbers near 80% start from math-pretrained models, or from bases that are already much higher than 37% before reinforcement learning.

Load `results/adapters/gsm8k-sft-v7` for the calculator. Do not load `v8` through `v14`.

## Run it

Apple Silicon, Python 3.12.

```bash
cd gsm8k_rl
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python scripts/prepare_gsm8k.py
python scripts/check_rewards.py
python scripts/benchmark.py --tag base --n 40
```

Calculator eval of a local adapter:

```bash
python scripts/benchmark.py --adapter results/adapters/gsm8k-sft-v7 --tools --n 200 --tag sft_v7_200
```

Adapters, the virtualenv, and the parquet dumps are gitignored. `prepare_gsm8k.py` rebuilds the data from the Hub. Training commands and the keep/discard outcomes are in the sections above. Plots in `report/figures/` are from the early GRPO runs (group collapse, KL, reward), not from `gsm8k-sft-v7`.

`HF_TOKEN` is optional for the public GSM8K dataset. Put it in `.env` (see `.env.example`). That file is gitignored.
