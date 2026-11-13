"""
hybrid.py -- the HYBRID categoriser: rules + AI model, each doing what it's best at.

  1. RULES first, but ONLY for clear-cut formats where a rule is (almost)
     never wrong:  ATM -> Cash,  refunds -> Refund,  dividends / salary /
     interest / cashback -> Income.
  2. Everything else (shops, merchants, people) -> the fine-tuned MODEL,
     which understands names it has never seen (82% vs 38% for rules).
  3. If the model gives an INVALID answer -> fall back to the full rules.

Bonus: rows handled by rules skip the slow model, so it's faster too.

Each answer also says WHERE it came from ("rule", "model", "fallback"),
so you can explain any decision. That's called traceability.
"""

import pandas as pd

from baseline_rules import learn_rules, predict as rules_predict
from parse_statement import classify


def clear_cut_rule(narration):
    """Return a category ONLY when the format makes it certain, else None."""
    n = narration.upper()
    typ, _ = classify(n)
    if typ == "ATM":
        return "Cash"
    if typ == "REFUND" or n.startswith("REV-") or "REFUND" in n:
        return "Refund"
    if n.startswith("ACH C-") or "SALARY" in n or "CASHBACK" in n or typ == "INTEREST":
        return "Income"
    return None


class Hybrid:
    def __init__(self, model_name="qwen-1.5b"):
        from categoriser import Categoriser          # imported here: loading the model is slow
        self.model = Categoriser(model_name)
        self.lookup, self.keywords = learn_rules(pd.read_csv("data/train.csv"))

    def predict(self, narrations, model_preds=None):
        """Returns two lists: categories, and where each one came from.
        model_preds: optional, the model's answers already computed (avoids running it twice)."""
        cats, sources = [None] * len(narrations), [None] * len(narrations)
        to_model = []
        for i, n in enumerate(narrations):
            c = clear_cut_rule(n)
            if c:
                cats[i], sources[i] = c, "rule"
            else:
                to_model.append(i)
        if to_model:
            preds = ([model_preds[i] for i in to_model] if model_preds
                     else self.model.predict([narrations[i] for i in to_model]))
            for i, p in zip(to_model, preds):
                if p == "INVALID":
                    cats[i], sources[i] = rules_predict(narrations[i], self.lookup, self.keywords), "fallback"
                else:
                    cats[i], sources[i] = p, "model"
        return cats, sources


if __name__ == "__main__":
    tests = ["ATW-512967XXXXXX5616-S1ANCE25-CHENNAI", "ACH C- ITC FINDIV 26 27-111111",
             "UPI-THANGAM TEA STALL-THANGAMTEA@OKSBI-SBIN0001234-123456789012-UPI"]
    for r in tests:
        print(f"{str(clear_cut_rule(r)):8s} <- {r}")
