"""
categoriser.py -- runs YOUR fine-tuned model on the laptop (CPU, no internet
needed after the first download, no data leaves the machine).

How it loads:
  1. Reads models/<name>/.../adapter_config.json to find which base model was used
     (e.g. Qwen/Qwen2.5-1.5B-Instruct)
  2. Loads that base model (downloaded from Hugging Face the FIRST time only,
     then cached on your laptop)
  3. Puts your LoRA add-on on top, and merges it in for speed

Use from other scripts:
    from categoriser import Categoriser
    cat = Categoriser("qwen-1.5b")
    cat.predict(["UPI-SWIGGY-...", "ATW-..."])   ->  ["Food", "Cash"]

Quick test:  python categoriser.py qwen-1.5b
"""

import json
import sys
import time
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from categories import LABELS

MODELS_DIR = Path("models")


def find_adapter(name):
    """Find the folder containing adapter_config.json under models/<name> (any depth)."""
    hits = list((MODELS_DIR / name).rglob("adapter_config.json"))
    if not hits:
        raise SystemExit(f"No adapter_config.json found under {MODELS_DIR / name}. "
                         f"Folders in models/: {[p.name for p in MODELS_DIR.iterdir()] if MODELS_DIR.exists() else 'none'}")
    return hits[0].parent


def prompt(narration):        # MUST match the prompt used in training (Colab Step 6)
    return f"Categorise this bank transaction.\nNarration: {narration}\nCategory:"


def to_label(text):
    text = text.strip().lower()
    for lab in LABELS:
        if text.startswith(lab.lower()):
            return lab
    return "INVALID"


class Categoriser:
    def __init__(self, name="qwen-1.5b"):
        folder = find_adapter(name)
        base = json.loads((folder / "adapter_config.json").read_text())["base_model_name_or_path"]
        # 0.5B fits in full precision (~2 GB). 1.5B uses half precision (~3 GB instead of ~6 GB)
        dtype = torch.float32 if "0.5B" in base else torch.bfloat16
        print(f"Loading {base} + your LoRA add-on from {folder} ({str(dtype).split('.')[-1]})...")
        print("  (first time: downloads the base model, 1-3 GB. Later runs load from cache.)")
        t0 = time.time()
        self.tok = AutoTokenizer.from_pretrained(folder)
        model = AutoModelForCausalLM.from_pretrained(base, dtype=dtype)
        self.model = PeftModel.from_pretrained(model, folder).merge_and_unload().eval()
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        self.tok.padding_side = "left"
        print(f"  ready in {time.time() - t0:.0f}s")

    @torch.no_grad()
    def predict(self, narrations, batch=8):
        out = []
        t0 = time.time()
        for i in range(0, len(narrations), batch):
            if len(narrations) > batch:          # progress for long runs
                done = len(out)
                eta = (time.time() - t0) / done * (len(narrations) - done) / 60 if done else 0
                print(f"  categorising {done}/{len(narrations)}" + (f"  (~{eta:.0f} min left)" if done else ""), flush=True)
            enc = self.tok([prompt(n) for n in narrations[i:i + batch]], return_tensors="pt", padding=True)
            gen = self.model.generate(**enc, max_new_tokens=6, do_sample=False,
                                      pad_token_id=self.tok.pad_token_id)
            texts = self.tok.batch_decode(gen[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)
            out += [to_label(t) for t in texts]
        return out


if __name__ == "__main__":
    name = sys.argv[1] if len(sys.argv) > 1 else "qwen-1.5b"
    cat = Categoriser(name)
    examples = ["UPI-AMMA MESS-AMMAMESS@OKSBI-SBIN0001234-123456789012-UPI",
                "POS 512967XXXXXX1234 SRI KRISHNA SWEETS CHENNAI",
                "ACH D- SBI MUTUAL FUND MF SIP-322137",
                "ATW-512967XXXXXX5616-S1ANCE25-CHENNAI"]
    t0 = time.time()
    preds = cat.predict(examples)
    print(f"\n{(time.time() - t0) / len(examples) * 1000:.0f} ms per narration on your CPU\n")
    for p, n in zip(preds, examples):
        print(f"  {p:13s} <- {n}")
