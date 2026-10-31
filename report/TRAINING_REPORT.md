# Training report

Current numbers live in the repository README. This file is the short scorecard. The figures next to it are from the early GRPO runs, before the calculator traces that produced the kept adapter.

## Scorecard

Same model throughout: `mlx-community/Llama-3.2-1B-Instruct-4bit`. Test split never trained on. Calculator score is the last real `calc()` value. Text score is exact match on `<answer>` or `####`.

| Checkpoint | Protocol | Score | n | Status |
|---|---|---|---|---|
| No adapter | text, original prompt | 0.20 | 40 | Reference |
| No adapter | text, original prompt | 0.17 | 200 | Reference |
| `gsm8k-sft-v3` | text, original prompt | 0.275 | 40 | Best original-prompt adapter |
| `gsm8k-sft-v3` | text, original prompt | 0.25 | 200 | Best original-prompt adapter |
| No adapter | calculator | 0.00 | 200 | Reference |
| `gsm8k-tool-v3` | calculator | 0.19 | 200 | Superseded |
| `gsm8k-sft-v6` | calculator | 0.365 (73/200) | 200 | Superseded |
| **`gsm8k-sft-v7`** | **calculator** | **0.380 (76/200)** | **200** | **Kept** |
| **`gsm8k-sft-v7`** | **calculator, full test** | **0.374 (493/1319)** | **1,319** | **Overall number** |

80% of the test set is 1,056 correct. This run is at 493. It is ongoing.

Later adapters (`gsm8k-sft-v8`, `gsm8k-rl-v9`, `gsm8k-sft-v10`, `gsm8k-sft-v11`, the v12–v14 probes, and formula search) scored at or below the kept checkpoint, or were stopped on a train probe before the test split was touched. They are not loaded. The README table lists each one.

## Figures

`report/figures/` is the first GRPO attempt: policy loss, KL, reward mean and spread, all-equal groups, format vs task reward, length, and the 40-question drop from 0.20 to 0.10. That adapter is `results/failed_20step/`. It is not the checkpoint behind the 493/1,319 number.

## 4. Run log (append only)
