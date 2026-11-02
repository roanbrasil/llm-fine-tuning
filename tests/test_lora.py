"""LoRA claims from the article, checked on real tensors."""

import torch
from torch import nn

from finetune.lora import LoRALinear, lora_param_count, trainable_params


def test_parameter_count_matches_article():
    # "With r = 16 on a 4096x4096 matrix ... 131k parameters instead of 16.7M — about 0.8%"
    n = lora_param_count(4096, 4096, 16)
    assert n == 131_072
    assert 4096 * 4096 == 16_777_216
    assert round(100 * n / 4096**2, 1) == 0.8


def test_step_zero_equals_base_model():
    torch.manual_seed(0)
    base = nn.Linear(64, 64, bias=False)
    layer = LoRALinear(base, r=8)
    x = torch.randn(4, 64)
    assert torch.equal(layer(x), base(x))  # B starts at zero


def test_only_adapters_are_trainable():
    layer = LoRALinear(nn.Linear(128, 256), r=4)
    assert trainable_params(layer) == lora_param_count(128, 256, 4)
    assert not any(p.requires_grad for p in layer.base.parameters())


def test_training_moves_only_a_and_b():
    torch.manual_seed(0)
    layer = LoRALinear(nn.Linear(32, 32, bias=False), r=4)
    w0 = layer.base.weight.clone()
    opt = torch.optim.Adam([p for p in layer.parameters() if p.requires_grad], lr=1e-2)
    target = nn.Linear(32, 32, bias=False)
    x = torch.randn(256, 32)
    first = None
    for _ in range(200):
        loss = ((layer(x) - target(x).detach()) ** 2).mean()
        first = first if first is not None else loss.item()
        opt.zero_grad(); loss.backward(); opt.step()
    assert torch.equal(layer.base.weight, w0)
    assert loss.item() < first


def test_merge_is_exact():
    # "merge them by computing W + BA once" -> same outputs, no adapter overhead
    torch.manual_seed(0)
    layer = LoRALinear(nn.Linear(48, 48), r=8)
    with torch.no_grad():
        layer.B.normal_()
    x = torch.randn(5, 48)
    before = layer(x)
    layer.merge()
    after = layer(x)
    assert torch.allclose(before, after, atol=1e-5)
    assert torch.linalg.matrix_rank(layer.delta_weight()) <= 8
