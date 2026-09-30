"""Evaluates the full pipeline (page -> lines -> model) on hand-labelled snapshots of real product pages.

Compares the original line extractor with the current one, and picks the decision threshold on the dev pages
only; the test pages are used just for the final numbers.
Usage: python app/eval_real_pages.py   (needs the HTML snapshots in app/real_pages/html/)
"""
import json
import re
import sys
from pathlib import Path

from bs4 import BeautifulSoup

sys.path.insert(0, str(Path(__file__).parent))
import detector

HERE = Path(__file__).parent
LABELS = json.load(open(HERE / "real_pages" / "labels.json"))["pages"]
OUT = HERE.parent / "figures" / "real_pages_eval.json"
THRESHOLDS = [0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.97, 0.98, 0.99, 0.995]


def extract_v1(html):
    """The original extractor: every tag without block children became a line (kept here for comparison)."""
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "noscript", "svg", "template", "iframe", "head"]):
        tag.decompose()
    lines, seen = [], set()
    for el in soup.find_all(True):
        if any(child.name in detector.BLOCK_TAGS for child in el.find_all(True)):
            continue
        text = re.sub(r"\s+", " ", el.get_text(" ", strip=True))
        if 2 <= len(text.split()) <= 40 and text not in seen:
            seen.add(text)
            lines.append(text)
    return lines[:detector.MAX_SNIPPETS]


def extract_v2(html):
    return detector.extract_snippets(html)[1]


def score(pages, threshold):
    """Line-level precision (are flagged lines real dark patterns?) and label-level recall (were they all found?)."""
    flagged = true_pos = found = total = 0
    for page in pages:
        labels = list(page["dark"])
        total += len(labels)
        hits = [t for t, p in zip(page["lines"], page["probs"]) if p > threshold]
        # A flagged line counts as correct if it overlaps a labelled dark line (so old fragments get credit too)
        match = lambda line: [l for l in labels if l in line or line in l]
        flagged += len(hits)
        true_pos += sum(1 for h in hits if match(h))
        found += len({l for h in hits for l in match(h)})
    precision = true_pos / flagged if flagged else 1.0
    recall = found / total if total else 1.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"threshold": threshold, "flagged": flagged, "false_alarms": flagged - true_pos,
            "false_alarms_per_page": round((flagged - true_pos) / len(pages), 1),
            "precision": round(precision, 4), "recall": round(recall, 4), "f1": round(f1, 4)}


def main():
    model = detector.Detector()
    runs = {}
    for name, extract in (("v1", extract_v1), ("v2", extract_v2)):
        pages = []
        for page_id, info in LABELS.items():
            lines = extract((HERE / "real_pages" / "html" / f"{page_id}.html").read_text())
            pages.append({"id": page_id, "split": info["split"], "dark": info["dark"],
                          "lines": lines, "probs": model.dark_probabilities(lines)})
        runs[name] = pages

    split = lambda pages, s: [p for p in pages if p["split"] == s]
    sweep = [score(split(runs["v2"], "dev"), t) for t in THRESHOLDS]
    # Highest dev F1; with so few labelled lines several thresholds tie, and then the lowest one wins so the
    # dashboard misses as few dark patterns as possible
    best = max(sweep, key=lambda r: (round(r["f1"], 3), -r["threshold"]))

    results = {
        "pages": {s: [p["id"] for p in split(runs["v2"], s)] for s in ("dev", "test")},
        "lines_per_page": {n: round(sum(len(p["lines"]) for p in runs[n]) / len(runs[n])) for n in runs},
        "dev_threshold_sweep": sweep,
        "chosen_threshold": best["threshold"],
        "test": {
            "old extractor, threshold 0.5": score(split(runs["v1"], "test"), 0.5),
            "new extractor, threshold 0.5": score(split(runs["v2"], "test"), 0.5),
            f"new extractor, threshold {best['threshold']}": score(split(runs["v2"], "test"), best["threshold"]),
        },
    }
    OUT.write_text(json.dumps(results, indent=2))

    print(f"Lines per page: old {results['lines_per_page']['v1']}, new {results['lines_per_page']['v2']}")
    print("\nDev pages, new extractor:")
    for r in sweep:
        print(f"  threshold {r['threshold']:<6} flagged {r['flagged']:>3}  false alarms {r['false_alarms']:>3}  "
              f"precision {r['precision']:.2f}  recall {r['recall']:.2f}  F1 {r['f1']:.2f}")
    print(f"\nChosen threshold: {best['threshold']}\n\nTest pages:")
    for name, r in results["test"].items():
        print(f"  {name:<32} flagged {r['flagged']:>3}  false alarms {r['false_alarms']:>3} "
              f"({r['false_alarms_per_page']}/page)  precision {r['precision']:.2f}  recall {r['recall']:.2f}  F1 {r['f1']:.2f}")
    for page in split(runs["v2"], "test"):
        missed = [l for l in page["dark"] if not any(p > best["threshold"] and (l in t or t in l)
                                                   for t, p in zip(page["lines"], page["probs"]))]
        wrong = [t for t, p in zip(page["lines"], page["probs"])
                 if p > best["threshold"] and not any(l in t or t in l for l in page["dark"])]
        if missed or wrong:
            print(f"  {page['id']}: missed {missed}  false alarms {wrong}")


if __name__ == "__main__":
    main()
