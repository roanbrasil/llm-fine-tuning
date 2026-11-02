"""Bradley-Terry and DPO losses behave as the article describes."""

import math

import pytest
import torch

from finetune.preference import bradley_terry_loss, dpo_loss, sequence_logprob


def test_bradley_terry_prefers_higher_chosen_score():
    lo = bradley_terry_loss(torch.tensor([2.0]), torch.tensor([0.0]))
    hi = bradley_terry_loss(torch.tensor([0.0]), torch.tensor([2.0]))
    assert lo < hi
    assert bradley_terry_loss(torch.tensor([1.0]), torch.tensor([1.0])).item() == pytest.approx(math.log(2))


def test_dpo_starts_at_log2_when_policy_equals_reference():
    lp = torch.tensor([-12.0, -30.0])
    loss, rc, rr = dpo_loss(lp, lp - 3, lp, lp - 3)
    assert abs(loss.item() - math.log(2)) < 1e-6
    assert torch.all(rc == 0) and torch.all(rr == 0)


def test_dpo_is_relative_to_the_reference():
    # the chosen answer being *absolutely* less likely does not matter, only the change vs the reference
    loss, _, _ = dpo_loss(torch.tensor([-50.0]), torch.tensor([-10.0]),
                          torch.tensor([-52.0]), torch.tensor([-8.0]))
    assert loss.item() < math.log(2)


def test_dpo_gradient_raises_chosen_and_lowers_rejected():
    pc = torch.tensor([-10.0], requires_grad=True)
    pr = torch.tensor([-10.0], requires_grad=True)
    loss, _, _ = dpo_loss(pc, pr, torch.tensor([-10.0]), torch.tensor([-10.0]))
    loss.backward()
    assert pc.grad.item() < 0 < pr.grad.item()  # gradient descent pushes chosen up, rejected down


def test_beta_controls_how_far_you_move():
    args = (torch.tensor([-9.0]), torch.tensor([-11.0]), torch.tensor([-10.0]), torch.tensor([-10.0]))
    small, _, _ = dpo_loss(*args, beta=0.01)
    big, _, _ = dpo_loss(*args, beta=1.0)
    assert big < small  # same shift counts for more with a larger beta


def test_sequence_logprob_masks_the_prompt():
    torch.manual_seed(0)
    logits = torch.randn(1, 6, 10)
    labels = torch.tensor([[-100, -100, -100, 4, 7, 1]])
    full = torch.log_softmax(logits, -1)
    expected = full[0, 2, 4] + full[0, 3, 7] + full[0, 4, 1]
    assert torch.allclose(sequence_logprob(logits, labels), expected.unsqueeze(0))
