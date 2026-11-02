"""Reproduces the pitfall we hit with the raw base model (Qwen/Qwen2.5-0.5B):

the ChatML end-of-turn token <|im_end|> is practically untrained in the base model, and its lm_head row is tied
to the frozen embedding. LoRA on the transformer layers alone cannot push p(<|im_end|>) past ~3e-4, so the
fine-tuned model writes the right JSON and then never stops. Making only the two ChatML embedding rows trainable
(PEFT `trainable_token_indices`, 1,792 extra parameters) fixes it within ~20 steps.

Plain PyTorch loop on a fixed batch of 16 SFT examples; prints p(<|im_end|>) at the position it should be emitted.
"""

import torch
from peft import LoraConfig, get_peft_model

from common import RESULTS, load, save_json, tokenizer
from finetune.data import sft_records, splits
from train_sft_qlora import TARGETS

RAW_BASE = "Qwen/Qwen2.5-0.5B"


def batch_with_labels(tok, n=16):
    rows = sft_records(splits()["sft"][:n])
    ids, labels = [], []
    for r in rows:
        prompt = tok.apply_chat_template(r["prompt"], tokenize=False, add_generation_prompt=True)
        full = tok.apply_chat_template(r["prompt"] + r["completion"], tokenize=False)
        p = tok(prompt, add_special_tokens=False)["input_ids"]
        f = tok(full, add_special_tokens=False)["input_ids"]
        ids.append(f); labels.append([-100] * len(p) + f[len(p):])
    width = max(map(len, ids))
    pad = lambda xs, v: [x + [v] * (width - len(x)) for x in xs]
    input_ids = torch.tensor(pad(ids, tok.pad_token_id)).to(0)
    return {"input_ids": input_ids, "attention_mask": (torch.tensor(pad([[1] * len(x) for x in ids], 0))).to(0),
            "labels": torch.tensor(pad(labels, -100)).to(0)}


def run(tok, train_chat_tokens: bool, steps=60):
    kw = {"trainable_token_indices": {"embed_tokens": tok.convert_tokens_to_ids(["<|im_start|>", "<|im_end|>"])}} \
        if train_chat_tokens else {}
    model = get_peft_model(load(RAW_BASE, four_bit=True), LoraConfig(
        r=16, lora_alpha=32, task_type="CAUSAL_LM", target_modules=TARGETS, **kw))
    batch = batch_with_labels(tok)
    im_end = tok.convert_tokens_to_ids("<|im_end|>")
    pos = (batch["labels"] == im_end).nonzero().tolist()
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=2e-4)
    curve = []
    model.train()
    for step in range(steps + 1):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = model(**batch)
        if step % 10 == 0:
            lp = torch.log_softmax(out.logits.float(), -1)
            p = torch.stack([lp[b, j - 1, im_end] for b, j in pos]).exp().mean().item()
            curve.append({"step": step, "loss": round(out.loss.item(), 4), "p_im_end": p})
            print(train_chat_tokens, curve[-1])
        out.loss.backward(); opt.step(); opt.zero_grad()
    extra = sum(p.numel() for p in model.parameters() if p.requires_grad)
    del model; torch.cuda.empty_cache()
    return {"trainable_params": extra, "curve": curve}


if __name__ == "__main__":
    tok = tokenizer(RAW_BASE)
    save_json("end_token_probe.json", {"model": RAW_BASE,
                                       "lora_only": run(tok, False), "lora_plus_chat_token_rows": run(tok, True)})
