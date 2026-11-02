"""Step 1 — instruction tuning (SFT) with QLoRA: 4-bit NF4 base + bf16 LoRA adapters on every linear layer.

Then merge the adapters into a *bf16* copy of the base (never into the 4-bit one — see the article).
"""

import time

import torch
from datasets import Dataset
from peft import LoraConfig, PeftModel
from trl import SFTConfig, SFTTrainer

from common import BASE_MODEL, RUNS, load, peak_vram_gb, save_json, tokenizer
from finetune.data import sft_records, splits

TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
LORA = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, task_type="CAUSAL_LM", target_modules=TARGETS)


def main():
    tok = tokenizer()
    data = Dataset.from_list(sft_records(splits()["sft"]))
    args = SFTConfig(
        output_dir=str(RUNS / "sft"), num_train_epochs=1, per_device_train_batch_size=16,
        learning_rate=2e-4, lr_scheduler_type="cosine", warmup_steps=0.05, logging_steps=10,
        bf16=True, max_length=256, completion_only_loss=True, save_strategy="no", report_to=[], seed=0,
        gradient_checkpointing=True)
    torch.cuda.reset_peak_memory_stats()
    model = load(BASE_MODEL, four_bit=True)
    trainer = SFTTrainer(model=model, args=args, train_dataset=data, processing_class=tok, peft_config=LORA)
    model = trainer.model
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    t0 = time.time()
    result = trainer.train()
    secs = time.time() - t0
    vram = peak_vram_gb()
    trainer.model.save_pretrained(RUNS / "sft" / "adapter")
    losses = [h["loss"] for h in trainer.state.log_history if "loss" in h]

    # merge into a bf16 copy of the base
    del trainer, model
    torch.cuda.empty_cache()
    merged = PeftModel.from_pretrained(load(BASE_MODEL), RUNS / "sft" / "adapter").merge_and_unload()
    merged.save_pretrained(RUNS / "sft" / "merged")
    tok.save_pretrained(RUNS / "sft" / "merged")

    save_json("train_sft.json", {
        "method": "SFT + QLoRA (NF4 base, LoRA r=16 alpha=32 on q,k,v,o,gate,up,down)",
        "base_model": BASE_MODEL, "examples": len(data), "epochs": 1,
        "trainable_params": trainable, "total_params_4bit_view": total,
        "train_loss_first_logged": losses[0], "train_loss_last_logged": losses[-1],
        "train_runtime_s": round(secs, 1), "peak_vram_gb": round(vram, 2),
        "adapter_size_mb": round(sum(f.stat().st_size for f in (RUNS / "sft" / "adapter").glob("*.safetensors")) / 1e6, 1),
        "final_train_loss": result.training_loss})


if __name__ == "__main__":
    main()
