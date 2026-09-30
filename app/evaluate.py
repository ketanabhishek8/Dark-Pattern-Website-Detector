"""Evaluates a saved model on the original test set and the held-out Indian-style set."""
import json
import sys
from pathlib import Path

import pandas as pd
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).parent))
import detector
from india_data import HOLDOUT

# The dataset is half dark patterns, so 0.5 is the right cut-off here (the dashboard uses detector.THRESHOLD)
DATASET_THRESHOLD = 0.5

model_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else detector.MODEL_DIR
detector.MODEL_DIR = model_dir
d = detector.Detector()

df = pd.read_csv("https://raw.githubusercontent.com/yamanalab/ec-darkpattern/master/dataset/dataset.tsv", sep="\t").dropna(subset=["text"])
df["category"] = df["Pattern Category"].replace({"Obstruction": "Other", "Sneaking": "Other", "Forced Action": "Other"})
_, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df["category"])


def scores(texts, labels):
    pred = [int(p > DATASET_THRESHOLD) for p in d.dark_probabilities(list(texts))]
    p, r, f, _ = precision_recall_fscore_support(labels, pred, average="binary", zero_division=0)
    return {"accuracy": accuracy_score(labels, pred), "precision": p, "recall": r, "f1": f,
            "false_positives": int(sum(1 for y, q in zip(labels, pred) if y == 0 and q == 1)),
            "false_negatives": int(sum(1 for y, q in zip(labels, pred) if y == 1 and q == 0))}


out = {"original_test": scores(test_df["text"], test_df["label"]),
       "india_holdout": scores([t for t, _, _ in HOLDOUT], [y for _, y, _ in HOLDOUT])}
print(json.dumps(out, indent=2))
