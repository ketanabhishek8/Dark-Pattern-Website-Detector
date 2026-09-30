"""Evaluates a saved model on the dataset's test split and on the held-out Indian-style lines.

Usage: python app/evaluate.py [MODEL_DIR] [--split group|random] [--save FILE]
"""
import argparse
import json
import sys
from pathlib import Path

from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support

sys.path.insert(0, str(Path(__file__).parent))
import detector
from data import load_dataset, split_dataset
from india_data import HOLDOUT

# The dataset is half dark patterns, so 0.5 is the right cut-off here (the dashboard uses detector.THRESHOLD)
DATASET_THRESHOLD = 0.5

parser = argparse.ArgumentParser()
parser.add_argument("model_dir", nargs="?", default=detector.MODEL_DIR)
parser.add_argument("--split", choices=["group", "random"], default="group")
parser.add_argument("--save", type=Path)
args = parser.parse_args()

d = detector.Detector(args.model_dir)
_, test_df = split_dataset(load_dataset(), args.split)


def binary_scores(texts, labels):
    probs, _ = d.predict(list(texts))
    pred = [int(p > DATASET_THRESHOLD) for p in probs]
    p, r, f, _ = precision_recall_fscore_support(labels, pred, average="binary", zero_division=0)
    return {"accuracy": accuracy_score(labels, pred), "precision": p, "recall": r, "f1": f,
            "false_positives": sum(1 for y, q in zip(labels, pred) if y == 0 and q == 1),
            "false_negatives": sum(1 for y, q in zip(labels, pred) if y == 1 and q == 0)}


dark = test_df[test_df.label == 1]
_, types = d.predict(list(dark["text"]))
out = {
    "split": args.split,
    "test_size": len(test_df),
    "original_test": binary_scores(test_df["text"], test_df["label"]),
    "category_on_dark_test": {"accuracy": accuracy_score(dark["category"], types),
                              "macro_f1": f1_score(dark["category"], types, average="macro")},
    "india_holdout": binary_scores([t for t, _, _ in HOLDOUT], [y for _, y, _ in HOLDOUT]),
}
print(json.dumps(out, indent=2))
if args.save:
    args.save.write_text(json.dumps(out, indent=2))
