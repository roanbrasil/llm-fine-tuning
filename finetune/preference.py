"""Preference losses from the article, written out by hand.

Bradley-Terry (RLHF reward model):  L = -log sigma(r(x, y_w) - r(x, y_l))
DPO:  L = -log sigma(beta * [log pi(y_w)/pi_ref(y_w) - log pi(y_l)/pi_ref(y_l)])
"""

import torch
import torch.nn.functional as F


def bradley_terry_loss(r_chosen: torch.Tensor, r_rejected: torch.Tensor) -> torch.Tensor:
    return -F.logsigmoid(r_chosen - r_rejected).mean()


def dpo_loss(policy_chosen: torch.Tensor, policy_rejected: torch.Tensor,
             ref_chosen: torch.Tensor, ref_rejected: torch.Tensor, beta: float = 0.1):
    """Inputs are summed log-probabilities of each response. Returns (loss, chosen_reward, rejected_reward).

    The implicit reward is beta * log(pi / pi_ref): DPO's "the reward is a function of the policy".
    """
    chosen_reward = beta * (policy_chosen - ref_chosen)
    rejected_reward = beta * (policy_rejected - ref_rejected)
    loss = -F.logsigmoid(chosen_reward - rejected_reward).mean()
    return loss, chosen_reward.detach(), rejected_reward.detach()


def sequence_logprob(logits: torch.Tensor, labels: torch.Tensor, ignore_index: int = -100) -> torch.Tensor:
    """Sum of log p(token) over the response tokens only (labels == ignore_index are masked out)."""
    logits = logits[:, :-1]
    labels = labels[:, 1:]
    mask = labels != ignore_index
    safe = labels.masked_fill(~mask, 0)
    logp = torch.log_softmax(logits.float(), dim=-1).gather(-1, safe.unsqueeze(-1)).squeeze(-1)
    return (logp * mask).sum(-1)
