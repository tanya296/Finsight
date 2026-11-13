"""
evaluate.py -- shared scoring code used by EVERY categoriser
(rules today, the big LLM today, our fine-tuned model on Day 3).
Same test files + same scoring = a fair comparison.
"""

from collections import Counter

import pandas as pd


def score(true, pred, name=""):
    """Print accuracy, per-category accuracy and the most common mistakes."""
    true, pred = list(true), list(pred)
    correct = sum(t == p for t, p in zip(true, pred))
    acc = correct / len(true)
    print(f"\n{name}: {correct}/{len(true)} correct = {acc:.1%} accuracy")

    # Per-category accuracy: where is it strong / weak?
    df = pd.DataFrame({"true": true, "ok": [t == p for t, p in zip(true, pred)]})
    per_cat = df.groupby("true")["ok"].mean()
    print("  Per category:", "  ".join(f"{c} {v:.0%}" for c, v in per_cat.sort_values().items()))

    # Most common mistakes (true -> predicted)
    mistakes = Counter((t, p) for t, p in zip(true, pred) if t != p)
    if mistakes:
        print("  Top mistakes:", ", ".join(f"{t}->{p} ({n})" for (t, p), n in mistakes.most_common(4)))
    return acc
