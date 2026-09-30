"""Loads the trained models, extracts text snippets from a web page, and classifies each snippet."""
import re
import time
from pathlib import Path
from urllib.parse import urlparse

import joblib
import requests
import torch
from bs4 import BeautifulSoup
from transformers import AutoModelForSequenceClassification, AutoTokenizer

MODEL_DIR = Path(__file__).parent / "model"
THRESHOLD = 0.5
MAX_SNIPPETS = 800
BLOCK_TAGS = {"div", "p", "li", "ul", "ol", "section", "article", "header", "footer", "nav", "aside", "main",
              "h1", "h2", "h3", "h4", "h5", "h6", "table", "tr", "td", "th", "form", "button", "label", "dd", "dt"}
SKIP_TAGS = ["script", "style", "noscript", "svg", "template", "iframe", "head"]
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/126.0 Safari/537.36",
    "Accept-Language": "en-IN,en;q=0.9",
}

CATEGORY_INFO = {
    "Scarcity": {"what": "Claims stock is running out or demand is high, so you buy before thinking.",
                 "ccpa": "False urgency"},
    "Urgency": {"what": "Puts a clock on the decision with timers or deadlines that may not be real.",
                "ccpa": "False urgency"},
    "Social Proof": {"what": "Uses other shoppers' activity to pressure you into following the crowd.",
                     "ccpa": "False urgency (false popularity)"},
    "Misdirection": {"what": "Steers your choice with guilt-tripping wording or a loaded default option.",
                     "ccpa": "Confirm shaming / interface interference"},
    "Other": {"what": "Hides costs, forces extra steps, or makes leaving harder than joining.",
              "ccpa": "Basket sneaking / forced action / subscription trap"},
}


class ScanError(Exception):
    """A problem the user can fix, with a message that says how."""


class Detector:
    def __init__(self):
        if not (MODEL_DIR / "distilbert").exists():
            raise SystemExit("Model not found. Run `python app/train.py` first.")
        self.device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR / "distilbert")
        self.bert = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR / "distilbert").to(self.device).eval()
        self.cat_model = joblib.load(MODEL_DIR / "category_model.joblib")

    @torch.no_grad()
    def dark_probabilities(self, texts, batch_size=64):
        probs = []
        for i in range(0, len(texts), batch_size):
            enc = self.tokenizer(texts[i:i + batch_size], truncation=True, padding=True, max_length=64,
                                 return_tensors="pt").to(self.device)
            probs += torch.softmax(self.bert(**enc).logits, dim=-1)[:, 1].cpu().tolist()
        return probs

    def analyze_snippets(self, snippets):
        start = time.perf_counter()
        probs = self.dark_probabilities(snippets)
        flagged_idx = [i for i, p in enumerate(probs) if p > THRESHOLD]
        categories = self.cat_model.predict([snippets[i] for i in flagged_idx]) if flagged_idx else []
        cat_of = dict(zip(flagged_idx, categories))
        items = [{"index": i, "text": t, "p_dark": round(p, 4), "category": cat_of.get(i)}
                 for i, (t, p) in enumerate(zip(snippets, probs))]
        counts = {c: 0 for c in CATEGORY_INFO}
        for c in categories:
            counts[c] += 1
        return {
            "snippets": items,
            "total": len(snippets),
            "flagged": len(flagged_idx),
            "category_counts": counts,
            "category_info": CATEGORY_INFO,
            "model_ms": round((time.perf_counter() - start) * 1000),
        }


def extract_snippets(html):
    """Returns the page title and its visible text, one snippet per innermost block element, in page order."""
    soup = BeautifulSoup(html, "html.parser")
    title = soup.title.get_text(strip=True) if soup.title else ""
    for tag in soup(SKIP_TAGS):
        tag.decompose()
    snippets, seen = [], set()
    for el in soup.find_all(True):
        if any(child.name in BLOCK_TAGS for child in el.find_all(True)):
            continue  # a deeper block element holds this text
        text = re.sub(r"\s+", " ", el.get_text(" ", strip=True))
        words = len(text.split())
        if not (2 <= words <= 40) or text in seen:
            continue
        seen.add(text)
        snippets.append(text)
        if len(snippets) >= MAX_SNIPPETS:
            break
    return title, snippets


def fetch_page(url):
    if not re.match(r"^https?://", url):
        url = "https://" + url
    host = urlparse(url).hostname
    if not host:
        raise ScanError("That doesn't look like a web address. Paste a full link, like https://example.com/product.")
    try:
        resp = requests.get(url, headers=HEADERS, timeout=15)
    except requests.RequestException:
        raise ScanError(f"Couldn't reach {host}. Check the link and your internet connection, then scan again.")
    if resp.status_code in (401, 403, 429, 503):
        raise ScanError(f"{host} blocked the scan (HTTP {resp.status_code}). "
                        "Open the page yourself, copy its text, and use Paste text instead.")
    if resp.status_code >= 400:
        raise ScanError(f"{host} returned HTTP {resp.status_code}. Check the link and scan again.")
    return url, host, resp.text
