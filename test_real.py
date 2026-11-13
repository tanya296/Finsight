"""
test_real.py -- THE REAL-DATA CHECK.

Compares your hand-labels (real/real_labels.csv) with:
  - the rules baseline (Day 2)
  - your fine-tuned model alone (Day 3)
  - the HYBRID: clear-cut rules first, model for the rest (hybrid.py)

Nothing is sent anywhere. The output shows real narrations, so don't
paste it publicly -- share only the accuracy lines.

Run:  python test_real.py qwen-1.5b      (or qwen-0.5b)
"""

import sys

import pandas as pd

from baseline_rules import predict as rules_predict
from categories import LABELS
from evaluate import score
from hybrid import Hybrid

if __name__ == "__main__":
    name = sys.argv[1] if len(sys.argv) > 1 else "qwen-1.5b"
    df = pd.read_csv("real/real_labels.csv", dtype=str).fillna("")

    bad = df[~df.my_category.isin(LABELS)]
    if len(bad):
        print("These rows have a missing or misspelt label (fix them in real/real_labels.csv):")
        for _, r in bad.iterrows():
            print(f"  '{r.my_category}'  <- {r.narration[:60]}")
        raise SystemExit(f"\nValid labels: {', '.join(LABELS)}")

    hybrid = Hybrid(name)                                   # loads the model once
    narr = list(df.narration)
    df["rules"] = [rules_predict(n, hybrid.lookup, hybrid.keywords) for n in narr]
    df["model"] = hybrid.model.predict(narr)
    df["hybrid"], df["source"] = hybrid.predict(narr, model_preds=list(df.model))

    print(f"\n=== REAL transactions ({len(df)}) ===")
    score(df.my_category, df.rules, "RULES on real data")
    score(df.my_category, df.model, f"MODEL ({name}) on real data")
    score(df.my_category, df.hybrid, "HYBRID on real data")
    print(f"  Hybrid used: {df.source.value_counts().to_dict()}")

    print("\nRow by row (OK = matches your label):")
    mark = lambda p, t: ("OK " if p == t else "X  ") + p
    for _, r in df.iterrows():
        print(f"  you={r.my_category:13s} rules={mark(r.rules, r.my_category):16s} "
              f"model={mark(r.model, r.my_category):16s} hybrid={mark(r.hybrid, r.my_category):16s}"
              f"({r.source:8s}) {r.narration[:35]}")
    df.to_csv(f"real/real_results_{name}.csv", index=False)
    print(f"\nSaved real/real_results_{name}.csv")
    print("Note: with only a few rows, each mistake moves accuracy a lot. Add more months to get a reliable number.")
