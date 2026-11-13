"""
build_notebook.py -- writes day3_finetune.ipynb (the Colab notebook).
Cells tagged 'colab' only work on Colab (GPU check, pip, upload, download);
cell tagged 'load' downloads the real model. test_notebook.py skips/replaces those.
"""

import json

CELLS = []


def md(text):
    CELLS.append({"cell_type": "markdown", "metadata": {}, "source": text.strip()})


def code(text, tag=None):
    CELLS.append({"cell_type": "code", "metadata": {"tags": [tag] if tag else []},
                  "execution_count": None, "outputs": [], "source": text.strip()})


md("""
# FinSight Day 3: Fine-tune a small AI model to categorise transactions

**Before you start:** menu **Runtime → Change runtime type → T4 GPU → Save**.

Then run each cell in order with **Shift+Enter**. Read the text above each cell: it explains what the cell does.

| Step | What happens | Time |
|---|---|---|
| 1–3 | Check GPU, install libraries, upload your data files | 3 min |
| 4–5 | Download the small model (Qwen2.5, 0.5B parameters) | 1 min |
| 6 | Test it BEFORE training (zero-shot) | 2 min |
| 7–9 | Fine-tune with LoRA on train.csv | 5–10 min |
| 10–11 | Test it AFTER training, look at mistakes | 2 min |
| 12 | Download the trained model + predictions | 1 min |
""")

md("""
## Step 1: Check we have a GPU
A GPU does thousands of calculations at once, which makes training ~50x faster than a laptop CPU.
You should see **Tesla T4** below. If you get an error, do: Runtime → Change runtime type → T4 GPU.
""")
code("""
import torch
assert torch.cuda.is_available(), "No GPU! Runtime -> Change runtime type -> T4 GPU, then run again."
print("GPU:", torch.cuda.get_device_name(0))
""", "colab")

md("""
## Step 2: Install libraries
- **transformers**: loads and runs AI models from Hugging Face
- **peft**: does LoRA (trains a small add-on instead of the whole model)
- **accelerate**: helps training run efficiently on the GPU
""")
code("""
!pip install -q -U transformers peft accelerate
!pip uninstall -y -q torchao   # Colab's old torchao clashes with new peft; we don't need it
""", "colab")

md("""
## Step 3: Upload your data files
(Or drag them into the **Files** panel on the left and skip this cell.)
Click **Choose Files** and select these 5 files from your `bank-agent` folder (hold Ctrl to pick several):
`data/train.csv`, `data/test_seen.csv`, `data/test_unseen.csv`, `categories.py`, `evaluate.py`

Only FAKE data is uploaded. Never upload the real statement.
""")
code("""
from google.colab import files
import os
uploaded = files.upload()
needed = ["train.csv", "test_seen.csv", "test_unseen.csv", "categories.py", "evaluate.py"]
missing = [f for f in needed if not os.path.exists(f)]
print("Missing:", missing if missing else "nothing, all good!")
""", "colab")

md("""
## Step 4: Settings
These are the **hyperparameters**: settings you choose before training.
- `MODEL_NAME`: the small model we start from. Qwen2.5-0.5B-Instruct = 0.5 billion parameters (the big AI on Day 2 had 120 billion!)
- `EPOCHS`: how many times the model reads the whole training set
- `LEARNING_RATE`: how big each learning step is (too big = unstable, too small = slow)
- `LORA_R`: size of the LoRA add-on (bigger = can learn more, but slower)
""")
code("""
MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"   # try "Qwen/Qwen2.5-1.5B-Instruct" for a bigger model
EPOCHS = 2
LEARNING_RATE = 2e-4
BATCH_SIZE = 16       # for the 1.5B model use BATCH_SIZE = 4 and GRAD_ACCUM = 4 (fits the T4's memory)
GRAD_ACCUM = 1        # gradient accumulation: update every N batches (acts like a batch of BATCH_SIZE x N)
LORA_R = 16
MAX_LEN = 96          # max tokens per example (narrations are short)

import time, re
import pandas as pd
import torch
from categories import CATEGORIES, LABELS
from evaluate import score

device = "cuda" if torch.cuda.is_available() else "cpu"
train = pd.read_csv("train.csv")
tests = {"test_seen": pd.read_csv("test_seen.csv"), "test_unseen": pd.read_csv("test_unseen.csv")}
print(f"train: {len(train)} rows, tests: {[len(t) for t in tests.values()]}, device: {device}")
""")

md("""
## Step 5: Download the model
The **tokenizer** cuts text into tokens (word pieces) and turns them into numbers.
The **model** is the neural network itself.
""")
code("""
from transformers import AutoTokenizer, AutoModelForCausalLM
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, dtype=torch.float32).to(device)
print(f"Loaded {MODEL_NAME}: {sum(p.numel() for p in model.parameters()) / 1e6:.0f} million parameters")
print("Example tokens:", tokenizer.tokenize("UPI-SWIGGY-SWIGGY8@YBL-HDFC0123456"))
""", "load")

md("""
## Step 6: Test BEFORE training (zero-shot)
We give the small model the list of categories and ask it to answer, like we did with the big AI on Day 2.
Expect a LOW score: a 0.5B model isn't smart enough to do this well without training.
This is our **before** picture.
""")
code("""
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

CAT_LIST = ", ".join(LABELS)

def zero_shot_prompt(narration):
    return (f"Categorise this Indian bank transaction into one of: {CAT_LIST}.\\n"
            f"Narration: {narration}\\nCategory:")

def finetune_prompt(narration):          # short: after training the model knows the labels
    return f"Categorise this bank transaction.\\nNarration: {narration}\\nCategory:"

def to_label(text):
    \"\"\"Turn the model's raw output into one of our labels (or INVALID).\"\"\"
    text = text.strip().lower()
    for lab in LABELS:
        if text.startswith(lab.lower()):
            return lab
    return "INVALID"

@torch.no_grad()
def predict(narrations, make_prompt, batch=32):
    model.eval()
    tokenizer.padding_side = "left"      # for generation, pad on the LEFT so the answer comes right after the prompt
    out = []
    for i in range(0, len(narrations), batch):
        prompts = [make_prompt(n) for n in narrations[i:i + batch]]
        enc = tokenizer(prompts, return_tensors="pt", padding=True).to(device)
        gen = model.generate(**enc, max_new_tokens=6, do_sample=False,
                             pad_token_id=tokenizer.pad_token_id)
        texts = tokenizer.batch_decode(gen[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)
        out += [to_label(t) for t in texts]
    return out

results = {}
for name, t in tests.items():
    preds = predict(list(t.narration), zero_shot_prompt)
    results[("Small model, zero-shot", name)] = score(t.category, preds, f"SMALL MODEL (before training) on {name}")
""")

md("""
## Step 7: Prepare the training examples
Each example becomes: **prompt + correct answer**, e.g.
`Categorise this bank transaction.\\nNarration: UPI-SWIGGY-...\\nCategory: Food`

**Important trick (label masking):** we only grade the model on the ANSWER part (" Food").
The prompt tokens get the label `-100`, which means "ignore when calculating the error".
Otherwise it would waste effort learning to repeat the prompt.
""")
code("""
class Examples(torch.utils.data.Dataset):
    def __init__(self, df):
        self.items = []
        for n, c in zip(df.narration, df.category):
            p_ids = tokenizer(finetune_prompt(n), add_special_tokens=False)["input_ids"]
            a_ids = tokenizer(" " + c, add_special_tokens=False)["input_ids"] + [tokenizer.eos_token_id]
            ids = (p_ids + a_ids)[:MAX_LEN]
            labels = ([-100] * len(p_ids) + a_ids)[:MAX_LEN]      # grade the answer only
            self.items.append({"input_ids": ids, "labels": labels})
    def __len__(self):
        return len(self.items)
    def __getitem__(self, i):
        return self.items[i]

def collate(batch):
    \"\"\"Pad a batch to the same length (pad on the RIGHT for training).\"\"\"
    L = max(len(b["input_ids"]) for b in batch)
    pad = tokenizer.pad_token_id
    return {
        "input_ids": torch.tensor([b["input_ids"] + [pad] * (L - len(b["input_ids"])) for b in batch]),
        "labels": torch.tensor([b["labels"] + [-100] * (L - len(b["labels"])) for b in batch]),
        "attention_mask": torch.tensor([[1] * len(b["input_ids"]) + [0] * (L - len(b["input_ids"])) for b in batch]),
    }

train_ds = Examples(train)
ex = train_ds[0]
print("Example input :", tokenizer.decode(ex["input_ids"]))
print("Graded part   :", tokenizer.decode([t for t in ex["labels"] if t != -100]))
""")

md("""
## Step 8: Add LoRA
Instead of changing all 500 million numbers in the model, LoRA adds small extra matrices next to
the attention layers and trains ONLY those. Look at the output: we train **under 2%** of the parameters.
That's why it's fast, cheap, and the saved add-on is only a few MB.
""")
code("""
from peft import LoraConfig, get_peft_model
lora = LoraConfig(r=LORA_R, lora_alpha=2 * LORA_R, lora_dropout=0.05, task_type="CAUSAL_LM",
                  target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])
model = get_peft_model(model, lora)
model.print_trainable_parameters()
""")

md("""
## Step 9: Train!
Watch the **loss** (the model's error). It should drop quickly, e.g. from ~3 to under 0.1.
One **step** = the model looks at one batch of 16 examples and adjusts slightly.
""")
code("""
from transformers import Trainer, TrainingArguments
total_steps = -(-len(train_ds) // (BATCH_SIZE * GRAD_ACCUM)) * EPOCHS     # steps = batches per epoch x epochs
args = TrainingArguments(
    output_dir="out", num_train_epochs=EPOCHS, per_device_train_batch_size=BATCH_SIZE,
    learning_rate=LEARNING_RATE, lr_scheduler_type="cosine", warmup_steps=max(1, int(0.05 * total_steps)),
    logging_steps=10, save_strategy="no", report_to="none",
    fp16=(device == "cuda"), remove_unused_columns=False, gradient_accumulation_steps=GRAD_ACCUM)
model.train()
t0 = time.time()
Trainer(model=model, args=args, train_dataset=train_ds, data_collator=collate).train()
print(f"Training took {(time.time() - t0) / 60:.1f} minutes")
""")

md("""
## Step 10: Test AFTER training
Same test files, same scoring as the rules and the big AI. This is the moment of truth!
""")
code("""
pred_store = {}
for name, t in tests.items():
    t0 = time.time()
    preds = predict(list(t.narration), finetune_prompt)
    ms = (time.time() - t0) / len(t) * 1000
    results[("Small model, fine-tuned", name)] = score(t.category, preds, f"FINE-TUNED on {name}")
    print(f"  Speed: {ms:.0f} ms per narration")
    pred_store[name] = preds
    t.assign(pred=preds).to_csv(f"pred_finetuned_{name}.csv", index=False)

# Fill in Day 2 numbers from your own runs so the table is complete
results[("Rules", "test_seen")], results[("Rules", "test_unseen")] = 0.907, 0.383
results[("Big AI (gpt-oss-120b)", "test_seen")], results[("Big AI (gpt-oss-120b)", "test_unseen")] = 0.867, 0.847
table = pd.Series(results).unstack().reindex(["Rules", "Big AI (gpt-oss-120b)", "Small model, zero-shot", "Small model, fine-tuned"])
print("\\n", (table * 100).round(1).astype(str) + "%")
""")

md("""
## Step 11: Look at the mistakes
Accuracy is one number; mistakes tell you WHY. Are they the model's fault, or are our labels ambiguous?
""")
code("""
t = tests["test_unseen"].assign(pred=pred_store["test_unseen"])
wrong = t[t.category != t.pred]
print(f"{len(wrong)} mistakes on test_unseen. Examples:")
for _, r in wrong.head(12).iterrows():
    print(f"  true={r.category:13s} pred={r.pred:13s} {r.narration[:70]}")
""")

md("""
## Step 12: Save and download
This downloads a zip with the **LoRA add-on** (a few MB, not the whole model) plus the prediction files.
Put the zip in your `bank-agent` folder. We'll use it to run the model on your laptop.
""")
code("""
from google.colab import files
model.save_pretrained("finsight-categoriser-lora")
tokenizer.save_pretrained("finsight-categoriser-lora")
!zip -qr day3_results.zip finsight-categoriser-lora pred_finetuned_test_seen.csv pred_finetuned_test_unseen.csv
files.download("day3_results.zip")
""", "colab")

md("""
## Bonus: try your own narrations
Type any narration (fake or made-up) and see what the model says.
""")
code("""
mine = ["UPI-AMMA MESS-AMMAMESS@OKSBI-SBIN0001234-123456789012-UPI",
        "POS 512967XXXXXX1234 SRI KRISHNA SWEETS CHENNAI",
        "UPI-RAVI KUMAR-RAVI99@OKAXIS-UTIB0001234-123456789012-PAYMENT"]
for n, p in zip(mine, predict(mine, finetune_prompt)):
    print(f"{p:13s} <- {n}")
""")

nb = {"cells": CELLS, "metadata": {"accelerator": "GPU", "colab": {"provenance": [], "gpuType": "T4"},
      "kernelspec": {"name": "python3", "display_name": "Python 3"}, "language_info": {"name": "python"}},
      "nbformat": 4, "nbformat_minor": 0}
json.dump(nb, open("day3_finetune.ipynb", "w"), indent=1)
print(f"Wrote day3_finetune.ipynb with {len(CELLS)} cells")
