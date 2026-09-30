"""Trains the dark pattern models (same setup as the notebook, plus Indian-style training lines) and saves them
to app/model/ for the dashboard."""
import random
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

from india_data import TRAIN as INDIA_TRAIN

URL = "https://raw.githubusercontent.com/yamanalab/ec-darkpattern/master/dataset/dataset.tsv"
OUT = Path(__file__).parent / "model"
SEED, EPOCHS, BATCH, LR, MAX_LEN = 42, 3, 16, 2e-5, 64

random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
print("Using device:", device)

df = pd.read_csv(URL, sep="\t").dropna(subset=["text"])
df["category"] = df["Pattern Category"].replace({"Obstruction": "Other", "Sneaking": "Other", "Forced Action": "Other"})
train_df, _ = train_test_split(df, test_size=0.2, random_state=SEED, stratify=df["category"])

# Domain adaptation: add Indian-style lines to the training set only (the test split above is untouched)
india = pd.DataFrame(INDIA_TRAIN, columns=["text", "label", "category"])
train_df = pd.concat([train_df, india], ignore_index=True)
print(f"Training on {len(train_df)} texts ({len(india)} Indian-style)")

# Category classifier (TF-IDF + Logistic Regression) on dark pattern texts only
dark = train_df[train_df.label == 1]
cat_model = make_pipeline(TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True),
                          LogisticRegression(max_iter=1000, C=10, class_weight="balanced"))
cat_model.fit(dark["text"], dark["category"])

# DistilBERT dark / not-dark classifier
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
enc = tokenizer(list(train_df["text"]), truncation=True, padding="max_length", max_length=MAX_LEN, return_tensors="pt")
loader = DataLoader(TensorDataset(enc["input_ids"], enc["attention_mask"], torch.tensor(train_df["label"].values)),
                    batch_size=BATCH, shuffle=True)
bert = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=2).to(device)
optimizer = torch.optim.AdamW(bert.parameters(), lr=LR)
scheduler = get_linear_schedule_with_warmup(optimizer, 0, EPOCHS * len(loader))
bert.train()
for epoch in range(EPOCHS):
    total = 0
    for ids, mask, y in loader:
        optimizer.zero_grad()
        loss = bert(input_ids=ids.to(device), attention_mask=mask.to(device), labels=y.to(device)).loss
        loss.backward(); optimizer.step(); scheduler.step()
        total += loss.item()
    print(f"Epoch {epoch + 1}/{EPOCHS}  loss={total / len(loader):.4f}")

OUT.mkdir(exist_ok=True)
bert.save_pretrained(OUT / "distilbert")
tokenizer.save_pretrained(OUT / "distilbert")
joblib.dump(cat_model, OUT / "category_model.joblib")
print("Saved models to", OUT)
