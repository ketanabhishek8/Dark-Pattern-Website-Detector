"""Trains the dark pattern model and saves it to app/model/ for the dashboard.

One DistilBERT model with 6 classes: "Not Dark Pattern" plus the five dark pattern types. It is trained on the
dataset's training split (split by website), Indian-style template lines, and lines mined from real product pages.

Usage: python app/train.py [--split group|random] [--no-mined] [--out DIR]
"""
import argparse
import random
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

from data import LABELS, SEED, extra_training_data, load_dataset, split_dataset

EPOCHS, BATCH, LR, MAX_LEN = 3, 16, 2e-5, 64

parser = argparse.ArgumentParser()
parser.add_argument("--split", choices=["group", "random"], default="group")
parser.add_argument("--no-mined", action="store_true", help="leave out the lines mined from real pages")
parser.add_argument("--out", type=Path, default=Path(__file__).parent / "model")
args = parser.parse_args()

random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
print("Using device:", device)

train_df, _ = split_dataset(load_dataset(), args.split)
extra = extra_training_data(mined=not args.no_mined)
train_df = pd.concat([train_df[["text", "label", "category"]], extra], ignore_index=True)
label_id = {name: i for i, name in enumerate(LABELS)}
y = torch.tensor(train_df["category"].map(label_id).values)
print(f"Training on {len(train_df)} texts ({len(extra)} Indian-style/mined), split: {args.split}")
print(train_df["category"].value_counts().to_string())

tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
enc = tokenizer(list(train_df["text"]), truncation=True, padding="max_length", max_length=MAX_LEN, return_tensors="pt")
loader = DataLoader(TensorDataset(enc["input_ids"], enc["attention_mask"], y), batch_size=BATCH, shuffle=True)
bert = AutoModelForSequenceClassification.from_pretrained(
    "distilbert-base-uncased", num_labels=len(LABELS),
    id2label=dict(enumerate(LABELS)), label2id=label_id).to(device)
optimizer = torch.optim.AdamW(bert.parameters(), lr=LR)
scheduler = get_linear_schedule_with_warmup(optimizer, 0, EPOCHS * len(loader))
bert.train()
for epoch in range(EPOCHS):
    total = 0
    for ids, mask, labels in loader:
        optimizer.zero_grad()
        loss = bert(input_ids=ids.to(device), attention_mask=mask.to(device), labels=labels.to(device)).loss
        loss.backward(); optimizer.step(); scheduler.step()
        total += loss.item()
    print(f"Epoch {epoch + 1}/{EPOCHS}  loss={total / len(loader):.4f}")

if args.out.exists():
    shutil.rmtree(args.out)  # drop files from older model versions (e.g. the separate category model)
bert.save_pretrained(args.out / "distilbert")
tokenizer.save_pretrained(args.out / "distilbert")
print("Saved model to", args.out)
