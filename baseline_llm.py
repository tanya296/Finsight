"""
baseline_llm.py -- BASELINE 2: ask a BIG AI model (70 billion parameters).

No training at all -- we just describe the categories in a prompt and ask.
This is called "zero-shot". (Default model: OpenAI's open-weight
gpt-oss-120b on Groq -- 120 billion parameters.) Big models know that "DHABA" is food and
"BUNDL TECHNOLOGIES" is Swiggy, because they read most of the internet.

Downsides we'll measure: it's slow, needs internet, costs money at scale,
and you'd be sending bank data to someone else's server (fine here --
ONLY our FAKE test data is sent).

Setup (free): make an account at https://console.groq.com -> API Keys -> Create.
Run:  python baseline_llm.py          (it asks for the key; typing is hidden)
"""

import json
import os
import re
import time
import urllib.error
import urllib.request
from getpass import getpass

import pandas as pd

from categories import CATEGORIES, LABELS
from evaluate import score

API_URL = os.environ.get("LLM_BASE_URL", "https://api.groq.com/openai/v1") + "/chat/completions"
MODEL = os.environ.get("LLM_MODEL", "openai/gpt-oss-120b")
# If MODEL gets retired, try these (biggest/best first), else any available model
PREFERRED = ["openai/gpt-oss-120b", "moonshotai/kimi-k2-instruct", "qwen/qwen3-32b",
             "meta-llama/llama-4-maverick-17b-128e-instruct", "openai/gpt-oss-20b"]
BATCH = 25          # narrations per request (fewer requests = fewer rate limits)


def build_prompt(narrations):
    cats = "\n".join(f"- {c}: {d}" for c, d in CATEGORIES.items())
    items = "\n".join(f"{i + 1}. {n}" for i, n in enumerate(narrations))
    return (
        "You categorise Indian bank statement narrations.\n"
        f"Allowed categories:\n{cats}\n\n"
        "Narrations often contain a merchant name, UPI ID, bank code and reference numbers. "
        "Focus on WHO was paid. A person's name usually means Transfers.\n\n"
        f"Narrations:\n{items}\n\n"
        f"Reply with ONLY a JSON list of {len(narrations)} category names, in order. "
        'Example: ["Food", "Transfers"]'
    )


def available_models(key):
    req = urllib.request.Request(API_URL.replace("/chat/completions", "/models"), headers={
        "Authorization": f"Bearer {key}", "User-Agent": "finsight-student-project"})
    with urllib.request.urlopen(req, timeout=30) as r:
        return [m["id"] for m in json.loads(r.read())["data"]]


def pick_model(key):
    """Use MODEL if it exists, otherwise the best available alternative."""
    models = available_models(key)
    if MODEL in models:
        return MODEL
    for m in PREFERRED:
        if m in models:
            print(f"   ('{MODEL}' not available -> using '{m}')")
            return m
    chat = [m for m in models if not any(x in m for x in ("whisper", "tts", "guard", "embed"))]
    print(f"   ('{MODEL}' not available -> using '{chat[0]}'. All models: {models})")
    return chat[0]


def call_llm(prompt, key, retries=6):
    payload = {"model": MODEL, "temperature": 0,
               "messages": [{"role": "user", "content": prompt}]}
    if MODEL.startswith("openai/gpt-oss"):
        payload["reasoning_effort"] = "low"     # less "thinking" = faster, fewer tokens
    body = json.dumps(payload).encode()
    req = urllib.request.Request(API_URL, data=body, headers={
        "Authorization": f"Bearer {key}", "Content-Type": "application/json",
        "User-Agent": "finsight-student-project"})
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                return json.loads(r.read())["choices"][0]["message"]["content"]
        except urllib.error.HTTPError as e:
            if e.code == 429:                       # rate limited -> wait and retry
                wait = 10 * (attempt + 1)
                print(f"   rate limited, waiting {wait}s...")
                time.sleep(wait)
            elif e.code == 401 or e.code == 403:
                raise SystemExit("API key rejected (401). Check the key and try again.")
            else:
                raise SystemExit(f"API error {e.code}: {e.read().decode()[:300]}")
    raise SystemExit("Too many rate-limit errors. Wait a minute and run again.")


def parse_answer(text, n):
    """Pull the JSON list out of the reply; anything invalid becomes 'INVALID'."""
    m = re.search(r"\[.*\]", text, re.S)
    try:
        out = json.loads(m.group(0)) if m else []
    except json.JSONDecodeError:
        out = []
    out = [o if o in LABELS else "INVALID" for o in out]
    return (out + ["INVALID"] * n)[:n]          # always exactly n answers


def predict_all(narrations, key):
    preds = []
    for i in range(0, len(narrations), BATCH):
        chunk = narrations[i:i + BATCH]
        preds += parse_answer(call_llm(build_prompt(chunk), key), len(chunk))
        print(f"   {len(preds)}/{len(narrations)} done")
        time.sleep(2)                           # be gentle with the free tier
    return preds


if __name__ == "__main__":
    from env import load_env
    load_env()                                   # reads GROQ_API_KEY from .env if present
    key = os.environ.get("GROQ_API_KEY") or getpass("Paste your Groq API key (hidden): ").strip()
    # Check what we got WITHOUT printing the key itself
    print(f"   Got a key with {len(key)} characters, starting with '{key[:4]}'")
    if not key.startswith("gsk_"):
        raise SystemExit("That doesn't look like a Groq key (they start with 'gsk_'). "
                         "The paste probably didn't work -- try right-click to paste, then Enter.")
    try:
        MODEL = pick_model(key)
    except urllib.error.HTTPError as e:
        raise SystemExit(f"Could not list models (error {e.code}). Check your API key.")
    for split in ["test_seen", "test_unseen"]:
        test = pd.read_csv(f"data/{split}.csv")
        print(f"\nAsking {MODEL} about {len(test)} narrations in {split}...")
        t0 = time.time()
        test["pred"] = predict_all(list(test.narration), key)
        secs = time.time() - t0
        score(test.category, test.pred, f"BIG LLM on {split}")
        print(f"  Time: {secs:.0f}s total ({secs / len(test) * 1000:.0f} ms per narration, incl. waiting)")
        test.to_csv(f"data/pred_llm_{split}.csv", index=False)
