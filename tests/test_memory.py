"""VRAM numbers quoted in the article."""

import pytest

from finetune.memory import full_finetune_gb, llama_like_adapter_params, lora_gb, qlora_gb

N_7B = 7e9
# Llama-2-7B shape: 32 layers, d_model 4096, d_ff 11008
ADAPTERS_R16 = llama_like_adapter_params(32, 4096, 11008, r=16)


def test_full_finetune_7b_is_112_gb():
    # "roughly 16 bytes per parameter. A 7B model needs ~112 GB for optimizer state alone"
    assert full_finetune_gb(N_7B) == pytest.approx(112)


def test_bf16_checkpoint_is_14_gb():
    assert N_7B * 2 / 1e9 == pytest.approx(14)


def test_adapters_are_about_half_a_percent():
    assert ADAPTERS_R16 == 39_976_960
    assert ADAPTERS_R16 / N_7B < 0.01  # "Adam's state now covers ~1% of the model"


def test_lora_fits_24_gb_card_before_activations():
    gb = lora_gb(N_7B, ADAPTERS_R16)
    assert 14 < gb < 16  # ~20 GB in the article once activations are added
    assert gb < 24


def test_qlora_base_is_about_4_gb():
    # "A 7B drops from 14 GB to ~4 GB; a 70B from 140 GB to ~35 GB"
    assert qlora_gb(N_7B, 0) == pytest.approx(3.6, abs=0.2)
    assert qlora_gb(70e9, 0) == pytest.approx(36, abs=1.5)


def test_qlora_fits_8_gb_card_before_activations():
    assert qlora_gb(N_7B, ADAPTERS_R16) < 8
