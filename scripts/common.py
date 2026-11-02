"""Shared helpers: model loading (bf16 or 4-bit NF4), greedy evaluation, and preference scoring."""

import json
import math
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("HF_HOME", str(ROOT / ".hf-cache"))  # keep downloads inside the project

import torch  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig  # noqa: E402

from finetune.data import score  # noqa: E402
from finetune.preference import dpo_loss, sequence_logprob  # noqa: E402

# The instruct model already knows its ChatML template. With the raw base model, LoRA cannot learn to emit
# <|im_end|> and generation never stops -- see scripts/probe_end_token.py and the README.
BASE_MODEL = os.environ.get("BASE_MODEL", "Qwen/Qwen2.5-0.5B-Instruct")
RUNS = ROOT / "runs"
RESULTS = ROOT / "results"

NF4 = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
                         bnb_4bit_compute_dtype=torch.bfloat16)


def tokenizer(path: str = BASE_MODEL):
    tok = AutoTokenizer.from_pretrained(path)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok


def load(path: str, four_bit: bool = False):
    return AutoModelForCausalLM.from_pretrained(
        path, dtype=torch.bfloat16, device_map={"": 0}, quantization_config=NF4 if four_bit else None)


def chat_prompt(tok, question: str) -> str:
    return tok.apply_chat_template([{"role": "user", "content": question}], tokenize=False,
                                   add_generation_prompt=True)


def end_of_turn_ids(tok) -> list[int]:
    """Stop at the chat template's end-of-turn token (<|im_end|> for Qwen) as well as the plain EOS."""
    ids = {tok.eos_token_id}
    im_end = tok.convert_tokens_to_ids("<|im_end|>")
    if isinstance(im_end, int) and im_end != tok.unk_token_id:
        ids.add(im_end)
    return sorted(i for i in ids if i is not None)


@torch.no_grad()
def generate_eval(model, tok, rows: list[dict], batch: int = 32, show: int = 5) -> dict:
    model.eval()
    hits = {"format_ok": 0, "correct": 0}
    samples = []
    for i in range(0, len(rows), batch):
        chunk = rows[i:i + batch]
        enc = tok([chat_prompt(tok, r["question"]) for r in chunk], return_tensors="pt", padding=True).to(0)
        out = model.generate(**enc, max_new_tokens=32, do_sample=False, pad_token_id=tok.pad_token_id,
                             eos_token_id=end_of_turn_ids(tok))
        texts = tok.batch_decode(out[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)
        for r, t in zip(chunk, texts):
            s = score(t, r)
            for k in hits:
                hits[k] += s[k]
            if len(samples) < show:
                samples.append({"question": r["question"], "output": t.strip(), "expected": r["chosen"], **s})
    n = len(rows)
    return {"n": n, "format_ok": hits["format_ok"] / n, "correct": hits["correct"] / n, "samples": samples}


def _response_logprob(model, tok, question: str, answer: str) -> torch.Tensor:
    prompt = chat_prompt(tok, question)
    full = tok.apply_chat_template([{"role": "user", "content": question},
                                    {"role": "assistant", "content": answer}], tokenize=False)
    assert full.startswith(prompt)
    p_ids = tok(prompt, add_special_tokens=False)["input_ids"]
    ids = tok(full, add_special_tokens=False, return_tensors="pt")["input_ids"].to(0)
    labels = ids.clone()
    labels[:, :len(p_ids)] = -100  # mask the prompt: only the response counts
    return sequence_logprob(model(ids).logits, labels)


@torch.no_grad()
def preference_eval(policy, ref, tok, rows: list[dict], beta: float = 0.1) -> dict:
    """Held-out pairs: how often the policy's implicit reward ranks chosen above rejected, and the DPO loss."""
    policy.eval(); ref.eval()
    wins, losses, margins = 0, [], []
    for r in rows:
        pc = _response_logprob(policy, tok, r["question"], r["chosen"])
        pr = _response_logprob(policy, tok, r["question"], r["rejected"])
        rc = _response_logprob(ref, tok, r["question"], r["chosen"])
        rr = _response_logprob(ref, tok, r["question"], r["rejected"])
        loss, cw, rw = dpo_loss(pc, pr, rc, rr, beta)
        losses.append(loss.item()); margins.append((cw - rw).item())
        wins += int((pc > pr).item())  # does the policy itself prefer the right answer?
    n = len(rows)
    return {"n": n, "policy_prefers_chosen": wins / n, "dpo_loss": sum(losses) / n,
            "mean_reward_margin": sum(margins) / n}


@torch.no_grad()
def perplexity(model, tok, text: str, window: int = 512) -> float:
    """Forgetting check from the article: general-text perplexity before vs after fine-tuning."""
    model.eval()
    ids = tok(text, return_tensors="pt")["input_ids"][0]
    nll, count = 0.0, 0
    for i in range(0, len(ids) - 1, window):
        chunk = ids[i:i + window + 1].unsqueeze(0).to(0)
        if chunk.shape[1] < 2:
            break
        out = model(chunk, labels=chunk)
        nll += out.loss.item() * (chunk.shape[1] - 1)
        count += chunk.shape[1] - 1
    return math.exp(nll / count)


def save_json(name: str, obj) -> None:
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / name).write_text(json.dumps(obj, indent=2) + "\n")
    print(json.dumps({k: v for k, v in obj.items() if k != "samples"}, indent=2) if isinstance(obj, dict) else obj)


def peak_vram_gb() -> float:
    return torch.cuda.max_memory_allocated() / 1e9
