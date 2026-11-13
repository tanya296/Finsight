"""
env.py -- loads secrets from a local .env file into environment variables.

The .env file (in the bank-agent folder) looks like:
    GROQ_API_KEY=gsk_your_key_here

Why a .env file?
  - The key is NOT written in the code, so code can be shared safely.
  - .gitignore lists .env, so it never goes to GitHub.
  - You don't have to paste the key every time.
"""

import os
from pathlib import Path


def load_env(path=".env"):
    p = Path(path)
    if not p.exists():
        return
    for line in p.read_text(encoding="utf-8-sig").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        name, value = line.split("=", 1)
        os.environ.setdefault(name.strip(), value.strip().strip('"').strip("'"))
