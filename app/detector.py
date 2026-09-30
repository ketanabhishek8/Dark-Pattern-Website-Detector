"""Loads the trained models, extracts text snippets from a web page, and classifies each snippet."""
import ipaddress
import os
import re
import socket
import time
from pathlib import Path
from urllib.parse import urljoin, urlparse

import joblib
import requests
import torch
from bs4 import BeautifulSoup, NavigableString
from transformers import AutoModelForSequenceClassification, AutoTokenizer

MODEL_DIR = Path(__file__).parent / "model"
# On real pages only ~1-2% of lines are dark patterns (vs 50% in the training data), so 0.5 raises far too many
# false alarms. 0.97 was chosen on hand-labelled dev pages; see eval_real_pages.py and figures/real_pages_eval.json
THRESHOLD = 0.97
MAX_SNIPPETS = 800
MAX_PAGE_BYTES = 5_000_000
MAX_REDIRECTS = 5
# Set on the public deployment so the server can't be used to reach private or internal addresses
BLOCK_PRIVATE_URLS = os.environ.get("BLOCK_PRIVATE_URLS") == "1"
BLOCK_TAGS = {"div", "p", "li", "ul", "ol", "section", "article", "header", "footer", "nav", "aside", "main",
              "h1", "h2", "h3", "h4", "h5", "h6", "table", "tr", "td", "th", "form", "button", "label", "dd", "dt",
              "blockquote", "figcaption", "caption", "summary", "details", "dialog", "fieldset", "legend", "address"}
SKIP_TAGS = ["script", "style", "noscript", "svg", "template", "iframe", "head", "select", "option", "textarea"]
# Text a shopper never sees: screen-reader copies, collapsed popups, hidden helpers (generic names plus Amazon's)
HIDDEN_CLASSES = {"a-offscreen", "a-hidden", "aok-hidden", "a-popover-preload", "sr-only", "visually-hidden",
                  "screen-reader-text", "hidden", "d-none", "hide"}
HIDDEN_STYLE = re.compile(r"display\s*:\s*none|visibility\s*:\s*hidden", re.I)
# Customer reviews are written by shoppers, not the seller, so they can't be the seller's dark pattern
USER_CONTENT = re.compile(r"review|comment|testimonial|^rh_|^cr-", re.I)
MAX_WORDS = 60
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

    def analyze_snippets(self, snippets, threshold=THRESHOLD):
        start = time.perf_counter()
        probs = self.dark_probabilities(snippets)
        flagged_idx = [i for i, p in enumerate(probs) if p > threshold]
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


def _is_hidden(el):
    if el.has_attr("hidden") or el.get("aria-hidden") == "true":
        return True
    if HIDDEN_STYLE.search(el.get("style", "")):
        return True
    return any(c in HIDDEN_CLASSES for c in el.get("class", []))


def _is_user_content(el):
    names = [el.get("id", ""), el.get("data-hook", ""), *el.get("class", [])]
    return any(USER_CONTENT.search(n) for n in names if n)


def _tidy(text):
    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(r"\s+([.,!?%:;)\]])", r"\1", text)      # "28,999 ." -> "28,999."
    text = re.sub(r"([₹$(\[])\s+", r"\1", text)           # "₹ 499" -> "₹499"
    return re.sub(r"(\d)\.\s+(\d)", r"\1.\2", text)       # "28,999. 00" -> "28,999.00"


def _keep(text):
    if "${" in text or "{{" in text:                        # unrendered template placeholders
        return False
    real_words = re.findall(r"[A-Za-z]{2,}", text)
    return len(real_words) >= 2 and len(text.split()) <= MAX_WORDS


def extract_snippets(html):
    """Returns the page title and its visible lines of text in page order.

    Each line is the text that belongs directly to one block element (a paragraph, list item, table cell, ...),
    with inline tags like <span> and <b> joined into it, so a sentence is never split into fragments.
    """
    soup = BeautifulSoup(html, "html.parser")
    title = soup.title.get_text(strip=True) if soup.title else ""
    for tag in soup(SKIP_TAGS):
        tag.decompose()
    for el in [el for el in soup.find_all(True) if _is_hidden(el) or _is_user_content(el)]:
        el.extract()

    blocks = {}  # block element -> its text pieces, in page order
    for piece in soup.find_all(string=True):
        if type(piece) is not NavigableString or not piece.strip():
            continue  # skip comments, doctype and whitespace
        block = next((p for p in piece.parents if p.name in BLOCK_TAGS), soup)
        blocks.setdefault(id(block), []).append(piece)

    lines, seen = [], set()
    for pieces in blocks.values():
        text = _tidy(" ".join(pieces))
        if text not in seen and _keep(text):
            seen.add(text)
            lines.append(text)
    # Drop a line that repeats part of a nearby longer line (e.g. a tooltip echoing its sentence)
    lines = [t for i, t in enumerate(lines)
             if not any(t != o and t in o for o in lines[max(0, i - 3):i + 4])]
    return title, lines[:MAX_SNIPPETS]


def _check_public(host):
    try:
        infos = socket.getaddrinfo(host, None)
    except socket.gaierror:
        raise ScanError(f"Couldn't find {host}. Check the link and scan again.")
    for info in infos:
        ip = ipaddress.ip_address(info[4][0])
        if not ip.is_global:
            raise ScanError(f"{host} points to a private network address, which this server won't scan.")


def fetch_page(url):
    if not re.match(r"^https?://", url):
        url = "https://" + url
    host = urlparse(url).hostname
    if not host:
        raise ScanError("That doesn't look like a web address. Paste a full link, like https://example.com/product.")
    try:
        for _ in range(MAX_REDIRECTS + 1):
            if BLOCK_PRIVATE_URLS:
                _check_public(host)
            resp = requests.get(url, headers=HEADERS, timeout=15, allow_redirects=False, stream=True)
            if resp.is_redirect and resp.headers.get("location"):
                url = urljoin(url, resp.headers["location"])
                host = urlparse(url).hostname or host
                resp.close()
                continue
            break
        else:
            raise ScanError(f"{host} redirected too many times. Open the page yourself and use Paste text instead.")
        body = resp.raw.read(MAX_PAGE_BYTES, decode_content=True)
        resp.close()
    except requests.RequestException:
        raise ScanError(f"Couldn't reach {host}. Check the link and your internet connection, then scan again.")
    if resp.status_code in (401, 403, 429, 503):
        raise ScanError(f"{host} blocked the scan (HTTP {resp.status_code}). "
                        "Open the page yourself, copy its text, and use Paste text instead.")
    if resp.status_code >= 400:
        raise ScanError(f"{host} returned HTTP {resp.status_code}. Check the link and scan again.")
    # requests assumes Latin-1 when the server sends no charset, which garbles UTF-8 pages (e.g. the ₹ sign)
    encoding = resp.encoding if "charset" in resp.headers.get("content-type", "").lower() else "utf-8"
    return url, host, body.decode(encoding or "utf-8", errors="replace")
