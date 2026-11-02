import json

from finetune.data import (EVAL_SEED, PAIRS_SEED, TRAIN_SEED, dpo_records, make_split,
                           score, sft_records, splits)


def test_targets_are_correct():
    for r in make_split(500, TRAIN_SEED):
        assert score(r["chosen"], r) == {"format_ok": True, "correct": True}


def test_rejected_has_right_format_but_wrong_value():
    for r in make_split(500, PAIRS_SEED):
        assert score(r["rejected"], r) == {"format_ok": True, "correct": False}


def test_score_rejects_bad_format():
    r = make_split(1, EVAL_SEED)[0]
    assert not score("The answer is 3 miles.", r)["format_ok"]
    assert not score(json.dumps({"value": r["target"], "unit": "??"}), r)["format_ok"]
    assert not score(json.dumps({"value": r["target"], "unit": r["unit"], "note": 1}), r)["format_ok"]


def test_splits_are_reproducible_and_disjoint():
    assert make_split(50, EVAL_SEED) == make_split(50, EVAL_SEED)
    s = splits()
    assert s == splits()
    train = {r["question"] for r in s["sft"] + s["pairs"]}
    held = [r["question"] for r in s["eval"]]
    held_pairs = [r["question"] for r in s["eval_pairs"]]
    assert train.isdisjoint(held) and train.isdisjoint(held_pairs) and set(held).isdisjoint(held_pairs)
    assert len(set(held)) == len(held)


def test_trl_record_shapes():
    rows = make_split(3, TRAIN_SEED)
    s = sft_records(rows)[0]
    assert s["prompt"][0]["role"] == "user" and s["completion"][0]["role"] == "assistant"
    d = dpo_records(rows)[0]
    assert set(d) == {"prompt", "chosen", "rejected"}
