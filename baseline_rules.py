"""
baseline_rules.py -- BASELINE 1: the "no AI" approach.

A baseline is the simple method you must BEAT. If your fancy AI model
can't beat this, it isn't worth it.

How it works (what a developer would hand-write):
  1. Transaction-type rules  : ATM -> Cash, ACH C -> Income, IMPS/TPT -> Transfers ...
  2. Merchant lookup table   : built from train.csv  (e.g. SWIGGY -> Food)
  3. Keyword rules           : HOTEL -> Food, MEDICALS -> Health ...  (from train.csv)
  4. Nothing matched?        : guess "Transfers" (most UPI payments to unknown names)

Run:  python baseline_rules.py
"""

import re
from collections import Counter, defaultdict

import pandas as pd

from evaluate import score
from parse_statement import classify


def learn_rules(train):
    """Build the lookup table and keyword list from the training data."""
    lookup = defaultdict(Counter)        # counterparty -> category counts
    keywords = defaultdict(Counter)      # single word  -> category counts
    for narr, cat in zip(train.narration, train.category):
        _, who = classify(narr)
        if who:
            lookup[who][cat] += 1
            for word in who.split():
                keywords[word][cat] += 1
    lookup = {k: v.most_common(1)[0][0] for k, v in lookup.items()}
    # keep only words that almost always mean ONE category (e.g. MEDICALS)
    keywords = {w: c.most_common(1)[0][0] for w, c in keywords.items()
                if sum(c.values()) >= 3 and c.most_common(1)[0][1] / sum(c.values()) > 0.9}
    return lookup, keywords


def predict(narr, lookup, keywords):
    n = narr.upper()
    typ, who = classify(n)
    # 1. transaction-type rules
    if typ == "ATM":
        return "Cash"
    if typ in ("REFUND",) or n.startswith("REV-") or "REFUND" in n:
        return "Refund"
    if typ in ("INTEREST",) or n.startswith("ACH C-") or "SALARY" in n or "CASHBACK" in n:
        return "Income"
    if typ in ("IMPS", "TPT") or n.startswith("NEFT DR"):
        return "Transfers"
    # 2. exact merchant lookup
    if who in lookup:
        return lookup[who]
    # 3. keyword rules
    for word in re.split(r"[\s\-/]+", who or n):
        if word in keywords:
            return keywords[word]
    # 4. default guess
    return "Transfers"


if __name__ == "__main__":
    train = pd.read_csv("data/train.csv")
    lookup, keywords = learn_rules(train)
    print(f"Learned {len(lookup)} merchant lookups and {len(keywords)} keyword rules from train.csv")

    results = {}
    for split in ["test_seen", "test_unseen"]:
        test = pd.read_csv(f"data/{split}.csv")
        test["pred"] = [predict(n, lookup, keywords) for n in test.narration]
        results[split] = score(test.category, test.pred, f"RULES on {split}")
        test.to_csv(f"data/pred_rules_{split}.csv", index=False)

    print("\nExamples the rules got WRONG on test_unseen:")
    wrong = test[test.category != test.pred].head(6)
    for _, r in wrong.iterrows():
        print(f"  true={r.category:12s} pred={r.pred:12s} {r.narration[:70]}")
