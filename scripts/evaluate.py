"""Compare base -> SFT -> SFT+DPO on held-out prompts the models never saw in training."""

import torch
from peft import PeftModel

from common import BASE_MODEL, RUNS, generate_eval, load, perplexity, preference_eval, save_json, tokenizer
from finetune.data import splits


def general_text(n_chars: int = 60_000) -> str:
    """First ~60k characters of the WikiText-2 test split: the forgetting check from the article."""
    from datasets import load_dataset
    rows = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")["text"]
    return "".join(rows)[:n_chars]


def main():
    s = splits()
    general = general_text()  # general English prose, never used for training
    tok = tokenizer()
    report = {}

    base = load(BASE_MODEL)
    report["base"] = {**generate_eval(base, tok, s["eval"]), "general_text_ppl": perplexity(base, tok, general)}
    del base; torch.cuda.empty_cache()

    sft = load(str(RUNS / "sft" / "merged"))
    report["sft"] = {**generate_eval(sft, tok, s["eval"]), "general_text_ppl": perplexity(sft, tok, general)}

    # the SFT model is also DPO's reference policy
    pref = {"sft": preference_eval(sft, sft, tok, s["eval_pairs"])}
    ref = load(str(RUNS / "sft" / "merged"))
    # DPO adapters on top of the bf16 SFT model, one variant at a time
    for name in ["dpo", "dpo_rpo", "dpo_on_policy", "dpo_on_policy_rpo"]:
        if not (RUNS / name / "adapter").exists():
            continue
        dpo = PeftModel.from_pretrained(load(str(RUNS / "sft" / "merged")), RUNS / name / "adapter")
        report[f"sft_{name}"] = {**generate_eval(dpo, tok, s["eval"]), "general_text_ppl": perplexity(dpo, tok, general)}
        pref[f"sft_{name}"] = preference_eval(dpo, ref, tok, s["eval_pairs"])
        del dpo; torch.cuda.empty_cache()
    report["held_out_pairs"] = pref
    save_json("eval.json", report)


if __name__ == "__main__":
    main()
