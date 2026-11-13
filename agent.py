"""
agent.py -- DAY 4: ask questions about your bank statement in plain English.

How it works (text-to-SQL):
  1. Your question + the table's COLUMN NAMES go to a big AI (Groq).
     The AI never sees your transactions.
  2. It replies with a SQL query + an answer sentence with blanks, e.g.
        SQL:      SELECT -SUM(amount) AS total FROM transactions WHERE category = 'Food' ...
        Sentence: "You spent Rs {total:,.2f} on Food."
  3. We CHECK the SQL is safe (read-only), then run it on YOUR laptop.
  4. The blanks in the sentence are filled in on your laptop.
  If the SQL has an error, the error message (not your data) goes back
  to the AI so it can fix it -- up to 2 retries.

Run:
  python agent.py                         (chat mode; type 'quit' to stop)
  python agent.py "How much on food?"     (one question)
"""

import json
import os
import re
import sqlite3
import sys
from getpass import getpass

import pandas as pd

from categories import LABELS

DB = "data/transactions.db"
MAX_ROWS = 20

SCHEMA = """Table: transactions   (one row per bank transaction)
  date            TEXT   'YYYY-MM-DD' (e.g. '2026-09-03')
  narration       TEXT   raw bank description, UPPERCASE (e.g. 'UPI-ZOMATO-...')
  withdrawal      REAL   money out, NULL if none
  deposit         REAL   money in, NULL if none
  amount          REAL   signed: NEGATIVE = money out, POSITIVE = money in
  closing_balance REAL   balance after this transaction
  txn_type        TEXT   one of: UPI, IMPS, NEFT, ATM, CARD, ACH, TPT, REFUND, GATEWAY, INTEREST, OTHER
  counterparty    TEXT   who was paid / who paid, UPPERCASE (may be empty)
  category        TEXT   one of: {cats}"""


def build_prompt(question, period, error=None, bad_sql=None):
    p = f"""You turn questions about a person's Indian bank statement into SQLite queries.

{SCHEMA.format(cats=", ".join(LABELS))}

The statement covers {period[0]} to {period[1]}. Money is in Indian Rupees.

Rules:
- Write ONE read-only SQLite SELECT query. No other statements.
- Spending = money out: use -SUM(amount) with amount < 0. Income/received = amount > 0.
- Dates: filter by date ONLY if the question mentions a time period (a month, "last month", a date).
  Months: date LIKE '2026-09%'. "Last month" = the latest month in the statement period.
- Names of people or shops: match with UPPER(narration) LIKE '%NAME%' (counterparty may be incomplete).
- Category values must be EXACTLY one of the listed categories.
- Give every computed column a short alias (e.g. AS total, AS n).
- Round money with ROUND(x, 2).
- Brand or shop names (Swiggy, Zepto, Amazon...): search the narration, don't rely on category alone.
- If the question is more specific than our categories (e.g. "clothes" vs the broader Shopping
  category, or "grocery apps" vs all Groceries), answer as best you can AND explain the
  approximation in "note".

Also write "answer": ONE short sentence for the user.
- If the query returns ONE row, put the result columns in curly braces, e.g.
  "You spent ₹{{total:,.2f}} on Food in September 2026."  ({{name}} or {{name:,.2f}} only)
- If it returns SEVERAL rows, write a short heading WITHOUT braces, e.g. "Spending by category:"

Also write "note": one short sentence stating any assumption or approximation you made
(e.g. "Treated 'clothes' as the whole Shopping category, which also includes electronics."),
or "" if the query matches the question exactly.

Reply with ONLY JSON: {{"sql": "...", "answer": "...", "note": "..."}}

Question: {question}"""
    if error:
        p += f"\n\nYour previous SQL failed.\nSQL: {bad_sql}\nError: {error}\nFix it."
    return p


# ---------------------------------------------------------------
# Safety: only read-only SELECT queries may run
# ---------------------------------------------------------------
FORBIDDEN = re.compile(r"\b(INSERT|UPDATE|DELETE|DROP|ALTER|CREATE|ATTACH|DETACH|PRAGMA|"
                       r"REPLACE|VACUUM|REINDEX|TRIGGER)\b", re.I)


def check_sql(sql):
    """Return the cleaned SQL, or raise ValueError if it isn't a single safe SELECT."""
    s = sql.strip().rstrip(";").strip()
    if ";" in s:
        raise ValueError("only one statement allowed")
    if not re.match(r"^(SELECT|WITH)\b", s, re.I):
        raise ValueError("only SELECT queries allowed")
    if FORBIDDEN.search(s):
        raise ValueError(f"forbidden keyword: {FORBIDDEN.search(s).group(0)}")
    return s


def run_sql(sql, db=DB):
    # mode=ro opens the database READ-ONLY: even a bad query can't change it
    with sqlite3.connect(f"file:{db}?mode=ro", uri=True) as con:
        return pd.read_sql(sql, con)


def fill(template, row):
    """Fill {col} / {col:,.2f} blanks from a result row (safe: only plain names allowed)."""
    def rep(m):
        name, spec = m.group(1), m.group(2) or ""
        if name not in row:
            raise KeyError(name)
        v = row[name]
        if v is None or (isinstance(v, float) and pd.isna(v)):
            v = 0
        try:
            return format(v, spec)
        except (ValueError, TypeError):
            return str(v)
    return re.sub(r"\{(\w+)(?::([^{}]*))?\}", rep, template)


def parse_reply(text):
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        raise ValueError("AI reply had no JSON")
    d = json.loads(m.group(0))
    return d["sql"], d.get("answer", ""), d.get("note", "")


# ---------------------------------------------------------------
# The agent
# ---------------------------------------------------------------
class Agent:
    def __init__(self, llm, db=DB):
        """llm: a function that takes a prompt string and returns the AI's reply text."""
        self.llm, self.db = llm, db
        with sqlite3.connect(f"file:{db}?mode=ro", uri=True) as con:
            cols = [r[1] for r in con.execute("PRAGMA table_info(transactions)")]
            if "category" not in cols:
                raise SystemExit("No 'category' column yet. Run: python add_categories.py qwen-1.5b")
            self.period = con.execute("SELECT MIN(date), MAX(date) FROM transactions").fetchone()

    def ask(self, question, retries=2):
        """Returns a dict: answer, sql, table (DataFrame), attempts, error."""
        error = sql = None
        for attempt in range(1, retries + 2):
            try:
                sql, template, note = parse_reply(self.llm(build_prompt(question, self.period, error, sql)))
                sql = check_sql(sql)
                df = run_sql(sql, self.db)
            except (ValueError, KeyError, json.JSONDecodeError, sqlite3.Error, pd.errors.DatabaseError) as e:
                error = str(e)
                continue
            if len(df) == 1:
                try:
                    answer = fill(template, df.iloc[0].to_dict())
                except KeyError:
                    answer = template + "\n" + df.to_string(index=False)
            elif len(df) == 0:
                answer = "No matching transactions found."
            else:
                answer = template.split("{")[0].strip() or "Results:"
            return {"answer": answer, "note": note, "sql": sql, "table": df.head(MAX_ROWS),
                    "attempts": attempt, "error": None}
        return {"answer": "Sorry, I couldn't answer that.", "note": "", "sql": sql, "table": None,
                "attempts": retries + 1, "error": error}


def groq_llm():
    """Connect to Groq (same key and model picker as Day 2's baseline_llm.py)."""
    import baseline_llm
    from env import load_env
    load_env()                                   # reads GROQ_API_KEY from .env if present
    key = os.environ.get("GROQ_API_KEY") or getpass("Paste your Groq API key (hidden): ").strip()
    if not key.startswith("gsk_"):
        raise SystemExit("That doesn't look like a Groq key (they start with 'gsk_').")
    baseline_llm.MODEL = baseline_llm.pick_model(key)
    print(f"Using {baseline_llm.MODEL}. The AI sees only your question + column names.\n")
    return lambda prompt: baseline_llm.call_llm(prompt, key)


def show(r):
    print(f"\n{r['answer']}")
    if r.get("note"):
        print(f"  Note: {r['note']}")
    if r["table"] is not None and len(r["table"]) > 1:
        print(r["table"].to_string(index=False))
    print(f"\n  SQL: {r['sql']}" + (f"   (fixed after {r['attempts'] - 1} retry)" if r["attempts"] > 1 else ""))
    if r["error"]:
        print(f"  Last error: {r['error']}")


if __name__ == "__main__":
    agent = Agent(groq_llm())
    if len(sys.argv) > 1:
        show(agent.ask(" ".join(sys.argv[1:])))
    else:
        print("Ask about your statement (type 'quit' to stop). Try: How much did we spend on food?")
        while (q := input("\nYou: ").strip()).lower() not in ("quit", "exit", "q"):
            if q:
                show(agent.ask(q))
