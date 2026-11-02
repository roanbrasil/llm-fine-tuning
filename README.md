# LLM fine-tuning, tested

Companion code for the article *Six Ways to Fine-Tune an LLM — and Why You Only Pick Two*. It checks the
article's claims in two ways:

1. **Unit tests** (`tests/`, CPU, ~2 s): LoRA, the VRAM arithmetic, and the Bradley–Terry / DPO losses,
   implemented by hand in `finetune/`.
2. **A real run** (`scripts/`, one GPU, ~5 min): the path the article recommends for a single GPU,
   **SFT with QLoRA, then DPO with QLoRA**, on `Qwen/Qwen2.5-0.5B-Instruct`, evaluated on held-out prompts.

Everything below was measured on an RTX 3080 (10 GB). Peak VRAM for every training run stayed under **0.8 GB**.

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate
pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
./run_all.sh           # tests + every training run + evaluation; JSON lands in results/
```

Models and datasets are cached in `.hf-cache/` inside the project.

```bash
# or one step at a time
cd scripts && python train_sft_qlora.py && python train_dpo_qlora.py --pairs on-policy --rpo && python evaluate.py
```

## The task

A procedurally generated unit-conversion task with a rigid JSON answer (`finetune/data.py`), so every
answer can be scored exactly:

```
user:      How many miles is 46.7 kilometers?
assistant: {"value": 29.02, "unit": "mi"}
```

- **SFT data:** 2,000 prompt → completion pairs; loss on the completion only.
- **DPO data:** 1,000 (prompt, chosen, rejected) pairs, in two flavours (see below).
- **Evaluation:** 300 held-out prompts and 300 held-out pairs, disjoint from training by construction (tested).
  An answer is *format-ok* if it is exactly `{"value": number, "unit": <right unit>}`, and *correct* if the
  value is within 1% (0.5 degrees for temperatures).
- **Forgetting check:** perplexity on the first 60k characters of WikiText-2 (test split).

## Results

| Model | Valid JSON | Correct value | WikiText-2 ppl |
| --- | ---: | ---: | ---: |
| Base (`Qwen2.5-0.5B-Instruct`) | 0.0% | 0.0% | 17.54 |
| + SFT (QLoRA) | 100.0% | 49.3% | 17.68 |
| + DPO, synthetic pairs | 100.0% | 39.7% | 18.44 |
| + DPO, synthetic pairs + SFT term (RPO) | 99.3% | 51.3% | 17.99 |
| + DPO, on-policy pairs | 96.7% | 23.7% | 18.69 |
| **+ DPO, on-policy pairs + SFT term (RPO)** | **99.7%** | **58.0%** | **17.81** |

One seed, 300 prompts: the standard error on each accuracy is about ±2.9 points.

The base model answers in prose ("To convert kilometers to miles, you can use the conversion factor..."). One epoch of SFT with QLoRA,
35 seconds, adapters of 17.6 MB, gives a model that never breaks the format. Perplexity on general text barely
moves (17.54 → 17.68), so forgetting stays small at this scale.

### What DPO taught us

The DPO rows are the interesting part, because two of them got *worse* while their training loss looked great.

- **Synthetic pairs** have a rejected answer with a mistake we invented: the inverted conversion factor, or a
  missing +32 offset. The SFT model already ranked the right answer first on 99.3% of held-out pairs, so there
  was nothing to learn. DPO's loss still fell from 0.60 to 0.03, and accuracy fell from 49.3% to 39.7%. The
  article's warning holds: *a falling loss doesn't mean generations got better.*
- **On-policy pairs** come from the SFT model's own wrong answers (`--pairs on-policy`: generate, score, keep
  the misses; 53% of its answers on fresh prompts were wrong). That is the "rounds" fix the article proposes
  for DPO being offline. On its own it was the worst run (23.7%). The logs show why: the implicit reward of
  the **chosen** answer went to −3.0. Chosen and rejected differ only in a few digit tokens, so pushing the
  rejected one down pulled the chosen one down with it.
- **Adding an SFT term on the chosen answer** (`--rpo`, TRL `loss_type=["sigmoid", "sft"]`) keeps the chosen
  likelihood anchored (its reward stays near 0). With on-policy pairs that gives the best model: **58.0%
  correct, up from 49.3% after SFT**.

So on this task, DPO helped only with pairs that cover the mistakes the model actually makes *and* an anchor
on the chosen answer.

## What broke: the base model never stops talking

The first attempt used the raw base model `Qwen/Qwen2.5-0.5B`, as in "turn a text completer into an
assistant". SFT learned the JSON perfectly, and then the model kept generating garbage after it: it never emitted
`<|im_end|>`, the ChatML end-of-turn token.

`scripts/probe_end_token.py` reproduces it on a fixed batch, measuring p(`<|im_end|>`) where it should be emitted
(`results/end_token_probe.json`):

| Steps | LoRA only | LoRA + trainable `<|im_start|>`/`<|im_end|>` rows |
| ---: | ---: | ---: |
| 0 | 0.000002 | 0.000002 |
| 20 | 0.00022 | 0.992 |
| 60 | 0.00031 | 0.9999 |

The base model barely saw the ChatML special tokens in pretraining, and its output head is tied to the
frozen embedding matrix. LoRA on the transformer layers alone cannot push that one logit up. Unfreezing just
those two embedding rows (PEFT `trainable_token_indices`, **1,792** extra parameters) fixes it within 20 steps.

That is the article's advice in practice: *use the template the base model already knows; inventing a new one
costs data.* Two practical notes:

- TRL 1.17's trainers compute the loss through a fused LM head that reads `lm_head.weight` directly, so it
  bypasses PEFT's trainable-token wrapper (it fails with `'TrainableTokensWrapper' object has no attribute
  'bias'`). That is why the main run uses the Instruct model, which already knows its template.
- Also stop generation on the template's end-of-turn token, not only on `eos_token`. For Qwen they differ
  (`<|im_end|>` vs `<|endoftext|>`); see `end_of_turn_ids` in `scripts/common.py`.

## Claims checked by the unit tests

| Claim in the article | Test |
| --- | --- |
| LoRA r=16 on 4096×4096 = 131,072 params, ≈0.8% of 16.7M | `test_lora.py::test_parameter_count_matches_article` |
| B starts at zero, so step 0 equals the base model | `test_lora.py::test_step_zero_equals_base_model` |
| Training moves only A and B; W stays frozen | `test_lora.py::test_training_moves_only_a_and_b` |
| Merging W + (α/r)BA gives identical outputs | `test_lora.py::test_merge_is_exact` |
| Full fine-tuning ≈ 16 bytes/param → 112 GB for a 7B | `test_memory.py::test_full_finetune_7b_is_112_gb` |
| QLoRA base: 7B → ~4 GB, 70B → ~35 GB | `test_memory.py::test_qlora_base_is_about_4_gb` |
| DPO starts at log 2 and only cares about the change vs the reference | `test_preference.py` |
| β controls how far the policy is allowed to move | `test_preference.py::test_beta_controls_how_far_you_move` |

## Layout

```
finetune/        lora.py (LoRA from scratch) · memory.py (VRAM math) · preference.py (BT + DPO losses) · data.py
scripts/         train_sft_qlora.py · train_dpo_qlora.py · evaluate.py · probe_end_token.py · common.py
tests/           22 unit tests
results/         JSON from the run above
make_figs.py     regenerates the article's diagrams (matplotlib)
```

## Setup used

RTX 3080 10 GB · torch 2.6.0 + CUDA 12.4 · transformers 5.22.1 · TRL 1.17.1 · PEFT 0.21.2 ·
bitsandbytes 0.50.2. QLoRA: NF4 + double quantization, bf16 compute, LoRA r=16 α=32 on
`q,k,v,o,gate,up,down`. SFT lr 2e-4; DPO lr 5e-5, β=0.1; one epoch each.
