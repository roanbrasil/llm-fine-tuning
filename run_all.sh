#!/usr/bin/env bash
# Full experiment on one GPU (~5 minutes on an RTX 3080, peak < 1 GB VRAM).
set -euo pipefail
cd "$(dirname "$0")"
python -m pytest -q
cd scripts
python probe_end_token.py
python train_sft_qlora.py
python train_dpo_qlora.py --pairs synthetic
python train_dpo_qlora.py --pairs synthetic --rpo
python train_dpo_qlora.py --pairs on-policy
python train_dpo_qlora.py --pairs on-policy --rpo
python evaluate.py
