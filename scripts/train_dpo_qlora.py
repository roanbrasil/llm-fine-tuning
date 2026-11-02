"""Step 2 — DPO with QLoRA on top of the merged SFT model.

Two kinds of preference pairs, to test a claim from the article ("if your pairs don't cover the kind of
mistake the model actually makes, DPO won't discover it on its own"):

  --pairs synthetic   rejected = a classic mistake we invented (inverted factor / missing +32 offset)
  --pairs on-policy   rejected = the SFT model's *own* wrong answers (generate, score, keep the misses)

The reference policy is the SFT model itself: with a PEFT model TRL gets it by disabling the adapters,
so only one copy of the weights sits in memory.
"""

import argparse
import time

import torch
from datasets import Dataset
from trl import DPOConfig, DPOTrainer

from common import RUNS, chat_prompt, end_of_turn_ids, load, peak_vram_gb, save_json, tokenizer
from finetune.data import PAIRS_SEED, assistant, dpo_records, make_split, score, splits, user
from train_sft_qlora import LORA

BETA = 0.1


@torch.no_grad()
def on_policy_pairs(sft_path: str, tok, n_pairs: int) -> tuple[list[dict], dict]:
    """Sample the SFT model's greedy answers on fresh training prompts and keep the wrong ones."""
    model = load(sft_path)
    model.eval()
    train_questions = {r["question"] for r in splits()["sft"]}
    pool = [r for r in make_split(6 * n_pairs, PAIRS_SEED + 100) if r["question"] not in train_questions]
    pairs, seen = [], 0
    for i in range(0, len(pool), 64):
        chunk = pool[i:i + 64]
        enc = tok([chat_prompt(tok, r["question"]) for r in chunk], return_tensors="pt", padding=True).to(0)
        out = model.generate(**enc, max_new_tokens=32, do_sample=False, pad_token_id=tok.pad_token_id,
                             eos_token_id=end_of_turn_ids(tok))
        texts = tok.batch_decode(out[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)
        for r, t in zip(chunk, texts):
            seen += 1
            if not score(t, r)["correct"]:
                pairs.append({"prompt": user(r["question"]), "chosen": assistant(r["chosen"]),
                              "rejected": assistant(t.strip())})
        if len(pairs) >= n_pairs:
            break
    del model; torch.cuda.empty_cache()
    return pairs[:n_pairs], {"prompts_sampled": seen, "sft_error_rate_on_pool": len(pairs) / seen}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", choices=["synthetic", "on-policy"], default="synthetic")
    ap.add_argument("--rpo", action="store_true",
                    help="add an SFT (NLL) term on the chosen answer: loss_type=[sigmoid, sft] (RPO)")
    a = ap.parse_args()
    sft_path = str(RUNS / "sft" / "merged")
    tok = tokenizer(sft_path)
    n = len(splits()["pairs"])
    extra = {}
    if a.pairs == "synthetic":
        records = dpo_records(splits()["pairs"])
    else:
        records, extra = on_policy_pairs(sft_path, tok, n)
    name = ("dpo" if a.pairs == "synthetic" else "dpo_on_policy") + ("_rpo" if a.rpo else "")
    args = DPOConfig(
        output_dir=str(RUNS / name), num_train_epochs=1, per_device_train_batch_size=8,
        learning_rate=5e-5, lr_scheduler_type="cosine", warmup_steps=0.05, logging_steps=10,
        loss_type=["sigmoid", "sft"] if a.rpo else ["sigmoid"],
        bf16=True, beta=BETA, max_length=256, save_strategy="no", report_to=[], seed=0,
        gradient_checkpointing=True)
    torch.cuda.reset_peak_memory_stats()
    trainer = DPOTrainer(model=load(sft_path, four_bit=True), args=args, train_dataset=Dataset.from_list(records),
                         processing_class=tok, peft_config=LORA)
    t0 = time.time()
    trainer.train()
    secs = time.time() - t0
    vram = peak_vram_gb()
    trainer.model.save_pretrained(RUNS / name / "adapter")
    logs = [h for h in trainer.state.log_history if "loss" in h]
    keys = [k for k in logs[-1] if k.startswith("rewards/") or k == "loss"]
    save_json(f"train_{name}.json", {
        "method": f"DPO{' + SFT term (RPO)' if a.rpo else ''} + QLoRA (beta={BETA}) on the merged SFT model, "
                  f"{a.pairs} pairs; reference = SFT model",
        "pairs": len(records), **extra, "example_pair": records[0], "epochs": 1,
        "train_runtime_s": round(secs, 1), "peak_vram_gb": round(vram, 2),
        "first_log": {k: logs[0].get(k) for k in keys}, "last_log": {k: logs[-1].get(k) for k in keys}})


if __name__ == "__main__":
    main()
