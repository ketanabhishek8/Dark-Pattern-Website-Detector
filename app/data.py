"""Training and evaluation data shared by train.py and evaluate.py, so both always use the same split."""
import json
from pathlib import Path

import pandas as pd
from sklearn.model_selection import GroupShuffleSplit, train_test_split

from india_data import TRAIN as INDIA_TRAIN

DATASET_URL = "https://raw.githubusercontent.com/yamanalab/ec-darkpattern/master/dataset/dataset.tsv"
MINED = Path(__file__).parent / "real_pages" / "mined.json"
# Class 0 must stay "Not Dark Pattern": P(dark) is computed as 1 - P(class 0)
LABELS = ["Not Dark Pattern", "Scarcity", "Urgency", "Social Proof", "Misdirection", "Other"]
SEED = 42
MINED_COPIES = 3  # the mined lines are few but come from exactly the pages the dashboard sees, so they count 3x


def load_dataset():
    df = pd.read_csv(DATASET_URL, sep="\t").dropna(subset=["text"])
    df["category"] = df["Pattern Category"].replace({"Obstruction": "Other", "Sneaking": "Other", "Forced Action": "Other"})
    return df


def split_dataset(df, how="group"):
    """80/20 split. "group" keeps each website's text on one side, so the test set only has unseen websites."""
    if how == "group":
        splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=SEED)
        train_idx, test_idx = next(splitter.split(df, groups=df["page_id"]))
        return df.iloc[train_idx], df.iloc[test_idx]
    return train_test_split(df, test_size=0.2, random_state=SEED, stratify=df["category"])


def extra_training_data(mined=True):
    """Indian-style template lines, plus hand-labelled lines mined from real product pages."""
    frames = [pd.DataFrame(INDIA_TRAIN, columns=["text", "label", "category"])]
    if mined:
        lines = pd.DataFrame(json.loads(MINED.read_text())["lines"])
        frames += [lines] * MINED_COPIES
    return pd.concat(frames, ignore_index=True)
