"""
eval_agent.py -- DAY 5: how often does the agent get the RIGHT answer?

Method: "execution accuracy" (how text-to-SQL systems are benchmarked, e.g. Spider).
  - 20 questions about the FAKE statement, each with a hand-written GOLD SQL query.
  - The agent answers each question; we run both queries and compare the RESULTS
    (not the SQL text: two different queries can both be right).
  - Uses the fake statement's TRUE categories, so categoriser mistakes don't
    count against the agent. We're testing the agent only.

Run:  python eval_agent.py          (needs GROQ_API_KEY in .env; ~1-2 minutes)
Output: score + every failure with both queries, saved to data/eval_results.csv
"""

import numbers
import sqlite3
import time

import pandas as pd

EVAL_DB = "data/eval.db"

# (question, gold SQL). Gold answers follow the agent's stated conventions:
# spending = money out (amount < 0); income/received = amount > 0.
QUESTIONS = [
    ("How much did I spend in total?",
     "SELECT ROUND(-SUM(amount),2) FROM transactions WHERE amount < 0"),
    ("How much money did I receive in total?",
     "SELECT ROUND(SUM(amount),2) FROM transactions WHERE amount > 0"),
    ("How much did I spend on food?",
     "SELECT ROUND(-SUM(amount),2) FROM transactions WHERE amount < 0 AND category = 'Food'"),
    ("How much did I spend on groceries in October 2026?",
     "SELECT ROUND(-SUM(amount),2) FROM transactions WHERE amount < 0 AND category = 'Groceries' AND date LIKE '2026-10%'"),
    ("How many transactions are there in the statement?",
     "SELECT COUNT(*) FROM transactions"),
    ("How many ATM withdrawals did I make?",
     "SELECT COUNT(*) FROM transactions WHERE txn_type = 'ATM'"),
    ("What was the biggest single withdrawal?",
     "SELECT ROUND(MAX(-amount),2) FROM transactions WHERE amount < 0"),
    ("How much did I spend on Swiggy?",
     "SELECT ROUND(-SUM(amount),2) FROM transactions WHERE amount < 0 AND UPPER(narration) LIKE '%SWIGGY%'"),
    ("How much did I spend on Amazon?",
     "SELECT ROUND(-SUM(amount),2) FROM transactions WHERE amount < 0 AND UPPER(narration) LIKE '%AMAZON%'"),
    ("What was my account balance at the end of the statement?",
     "SELECT closing_balance FROM transactions ORDER BY date DESC, rowid DESC LIMIT 1"),
    ("What is my average spend per withdrawal?",
     "SELECT ROUND(AVG(-amount),2) FROM transactions WHERE amount < 0"),
    ("How much did I spend in September 2026?",
     "SELECT ROUND(-SUM(amount),2) FROM transactions WHERE amount < 0 AND date LIKE '2026-09%'"),
    ("How much salary did I receive?",
     "SELECT ROUND(SUM(amount),2) FROM transactions WHERE amount > 0 AND UPPER(narration) LIKE '%SALARY%'"),
    ("How many UPI transactions were there?",
     "SELECT COUNT(*) FROM transactions WHERE txn_type = 'UPI'"),
    ("How much did I spend on travel and fuel combined?",
     "SELECT ROUND(-SUM(amount),2) FROM transactions WHERE amount < 0 AND category IN ('Travel','Fuel')"),
    ("What are my top 3 spending categories, excluding transfers?",
     "SELECT category FROM transactions WHERE amount < 0 AND category <> 'Transfers' "
     "GROUP BY category ORDER BY SUM(-amount) DESC LIMIT 3"),
    ("Which 3 payees did I spend the most money on?",
     "SELECT counterparty FROM transactions WHERE amount < 0 AND counterparty <> '' "
     "GROUP BY counterparty ORDER BY SUM(-amount) DESC LIMIT 3"),
    ("On which date did I spend the most money?",
     "SELECT date FROM transactions WHERE amount < 0 GROUP BY date ORDER BY SUM(-amount) DESC LIMIT 1"),
    ("How much did I spend on health in October 2026?",
     "SELECT ROUND(-SUM(amount),2) FROM transactions WHERE amount < 0 AND category = 'Health' AND date LIKE '2026-10%'"),
    ("How much money did I send to other people?",
     "SELECT ROUND(-SUM(amount),2) FROM transactions WHERE amount < 0 AND category = 'Transfers'"),
]


def build_eval_db():
    """Fake statement + its TRUE categories -> data/eval.db (read-only testing database)."""
    from parse_statement import parse_pdf
    df = parse_pdf("data/fake_statement.pdf")
    truth = pd.read_csv("data/fake_statement_truth.csv")
    df["category"] = truth.true_category.values
    with sqlite3.connect(EVAL_DB) as con:
        df.to_sql("transactions", con, if_exists="replace", index=False)


def values(df):
    """All cell values of a result, as comparable items (numbers rounded, text uppercased)."""
    out = []
    for v in df.to_numpy().ravel():
        if isinstance(v, numbers.Number) and not isinstance(v, bool) and not pd.isna(v):   # incl. numpy ints
            out.append(round(abs(float(v)), 2))
        elif isinstance(v, str):
            out.append(v.strip().upper())
    return out


def matches(gold, got):
    """Correct if every gold value appears in the agent's result.
    1 value  -> it must appear in the agent's FIRST row (a number within 0.01, or the exact text)
    N values -> all N must appear somewhere in the agent's result (order ignored)."""
    if got is None or len(got) == 0:
        return False
    g = values(gold)
    if not g:                       # never pass on an empty expected answer
        return False
    pool = values(got.head(1)) if len(g) == 1 else values(got)
    def found(x):
        return any((isinstance(x, float) and isinstance(y, float) and abs(x - y) <= 0.01) or x == y
                   for y in pool)
    return all(found(x) for x in g)


if __name__ == "__main__":
    from agent import Agent, groq_llm
    build_eval_db()
    agent = Agent(groq_llm(), db=EVAL_DB)
    rows, t0 = [], time.time()
    with sqlite3.connect(f"file:{EVAL_DB}?mode=ro", uri=True) as con:
        for i, (q, gold_sql) in enumerate(QUESTIONS, 1):
            gold = pd.read_sql(gold_sql, con)
            r = agent.ask(q)
            ok = matches(gold, r["table"])
            rows.append({"question": q, "correct": ok, "retries": r["attempts"] - 1,
                         "expected": ", ".join(map(str, values(gold))),
                         "got": "" if r["table"] is None else ", ".join(map(str, values(r["table"].head(3))))[:80],
                         "agent_sql": r["sql"], "gold_sql": gold_sql, "note": r.get("note", "")})
            print(f"{'OK ' if ok else 'X  '} {i:2d}. {q}")
            time.sleep(1)                     # gentle on the free API

    res = pd.DataFrame(rows)
    res.to_csv("data/eval_results.csv", index=False)
    n, k = len(res), int(res.correct.sum())
    print(f"\nEXECUTION ACCURACY: {k}/{n} = {k / n:.0%}   "
          f"({int((res.retries > 0).sum())} answers needed a self-correction retry; {time.time() - t0:.0f}s)")
    for _, r in res[~res.correct].iterrows():
        print(f"\nX  {r.question}\n   expected: {r.expected}\n   got     : {r.got}"
              f"\n   agent SQL: {r.agent_sql}\n   gold SQL : {r.gold_sql}" + (f"\n   note: {r.note}" if r.note else ""))
    print("\nSaved data/eval_results.csv")
