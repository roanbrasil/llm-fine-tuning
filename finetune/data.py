"""A small, procedurally generated task so every number in the experiment can be checked exactly.

Task: unit conversion with a rigid JSON answer, e.g.
    user:      Convert 12.5 kilometers to miles.
    assistant: {"value": 7.77, "unit": "mi"}

This is the kind of behaviour the article says fine-tuning is for: a rigid output format that a base
model will not produce on its own.

SFT examples are (prompt, completion). DPO pairs are (prompt, chosen, rejected), where the rejected
answer has the right format but a classic mistake: the inverted factor, or a missing temperature offset.
"""

import json
import random

# name, from, to, from_symbol(for prompt), to_symbol, forward, wrong
CONVERSIONS = [
    ("kilometers", "miles", "mi", lambda x: x * 0.621371, lambda x: x / 0.621371),
    ("miles", "kilometers", "km", lambda x: x * 1.609344, lambda x: x / 1.609344),
    ("kilograms", "pounds", "lb", lambda x: x * 2.204623, lambda x: x / 2.204623),
    ("pounds", "kilograms", "kg", lambda x: x / 2.204623, lambda x: x * 2.204623),
    ("meters", "feet", "ft", lambda x: x * 3.280840, lambda x: x / 3.280840),
    ("feet", "meters", "m", lambda x: x / 3.280840, lambda x: x * 3.280840),
    ("liters", "US gallons", "gal", lambda x: x * 0.264172, lambda x: x / 0.264172),
    ("US gallons", "liters", "L", lambda x: x * 3.785412, lambda x: x / 3.785412),
    ("degrees Celsius", "degrees Fahrenheit", "F", lambda x: x * 9 / 5 + 32, lambda x: x * 9 / 5),
    ("degrees Fahrenheit", "degrees Celsius", "C", lambda x: (x - 32) * 5 / 9, lambda x: x * 5 / 9),
]

TEMPLATES = [
    "Convert {v} {a} to {b}.",
    "How many {b} is {v} {a}?",
    "{v} {a} in {b}?",
    "Please express {v} {a} in {b}.",
]


def answer(value: float, unit: str) -> str:
    return json.dumps({"value": round(value, 2), "unit": unit})


def _example(rng: random.Random):
    a, b, unit, fwd, wrong = rng.choice(CONVERSIONS)
    v = round(rng.uniform(1, 500), rng.choice([0, 1]))
    v = int(v) if float(v).is_integer() else v
    prompt = rng.choice(TEMPLATES).format(v=v, a=a, b=b)
    return {"question": prompt, "x": v, "unit": unit,
            "target": round(fwd(v), 2), "chosen": answer(fwd(v), unit), "rejected": answer(wrong(v), unit)}


def make_split(n: int, seed: int) -> list[dict]:
    rng = random.Random(seed)
    return [_example(rng) for _ in range(n)]


def make_heldout(n: int, seed: int, exclude: list[dict]) -> list[dict]:
    """Like make_split, but drops any question that also appears in `exclude` (the training data)."""
    seen = {r["question"] for r in exclude}
    rng = random.Random(seed)
    rows = []
    while len(rows) < n:
        r = _example(rng)
        if r["question"] not in seen:
            seen.add(r["question"])
            rows.append(r)
    return rows


# experiment sizes
N_SFT, N_PAIRS, N_EVAL = 2000, 1000, 300


def splits() -> dict[str, list[dict]]:
    sft = make_split(N_SFT, TRAIN_SEED)
    pairs = make_split(N_PAIRS, PAIRS_SEED)
    held = make_heldout(N_EVAL, EVAL_SEED, sft + pairs)
    held_pairs = make_heldout(N_EVAL, PAIRS_EVAL_SEED, sft + pairs + held)
    return {"sft": sft, "pairs": pairs, "eval": held, "eval_pairs": held_pairs}


def user(text: str) -> list[dict]:
    return [{"role": "user", "content": text}]


def assistant(text: str) -> list[dict]:
    return [{"role": "assistant", "content": text}]


def sft_records(rows: list[dict]) -> list[dict]:
    """Conversational prompt-completion format: TRL computes the loss on the completion only."""
    return [{"prompt": user(r["question"]), "completion": assistant(r["chosen"])} for r in rows]


def dpo_records(rows: list[dict]) -> list[dict]:
    return [{"prompt": user(r["question"]), "chosen": assistant(r["chosen"]),
             "rejected": assistant(r["rejected"])} for r in rows]


def score(text: str, row: dict) -> dict:
    """format_ok: valid JSON with exactly {value: number, unit: <right unit>}; correct: within 1% (or 0.5 deg)."""
    text = text.strip()
    try:
        obj = json.loads(text)
        ok = (isinstance(obj, dict) and set(obj) == {"value", "unit"}
              and isinstance(obj["value"], (int, float)) and obj["unit"] == row["unit"])
    except (json.JSONDecodeError, TypeError):
        ok = False
    correct = False
    if ok:
        tol = 0.5 if row["unit"] in ("F", "C") else max(0.01, abs(row["target"]) * 0.01)
        correct = abs(obj["value"] - row["target"]) <= tol
    return {"format_ok": ok, "correct": correct}


# fixed seeds so train / eval never overlap by construction of the seed, and are reproducible
TRAIN_SEED, EVAL_SEED, PAIRS_SEED, PAIRS_EVAL_SEED = 1, 2, 3, 4
