"""Dark Pattern Detector dashboard. Run: python app/server.py, then open http://localhost:8000"""
import os
import time
from pathlib import Path
from urllib.parse import urlparse

import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse
from pydantic import BaseModel

from detector import Detector, ScanError, extract_snippets, fetch_page

STATIC = Path(__file__).parent / "static"
app = FastAPI(title="Dark Pattern Detector")
detector = Detector()


class ScanRequest(BaseModel):
    url: str | None = None
    text: str | None = None


@app.get("/")
def index():
    return FileResponse(STATIC / "index.html")


@app.get("/sample-shop")
def sample_shop():
    return FileResponse(STATIC / "sample-shop.html")


@app.post("/api/scan")
def scan(req: ScanRequest, request: Request):
    start = time.perf_counter()
    try:
        if req.url and req.url.strip():
            target = urlparse(req.url.strip())
            if target.path == "/sample-shop" and target.hostname == request.url.hostname:
                # Our own sample page: read it from disk instead of fetching it over the network
                url, host, html = req.url.strip(), target.hostname, (STATIC / "sample-shop.html").read_text()
            else:
                url, host, html = fetch_page(req.url.strip())
            title, snippets = extract_snippets(html)
            source = {"kind": "url", "url": url, "host": host, "title": title or host}
        elif req.text and req.text.strip():
            snippets = list(dict.fromkeys(l.strip() for l in req.text.splitlines() if l.strip()))
            source = {"kind": "text", "title": "Pasted text"}
        else:
            raise ScanError("Paste a link or some page text to scan.")
        if not snippets:
            raise ScanError("No readable text found on that page. It may build its content with JavaScript. "
                            "Copy the page text and use Paste text instead.")
    except ScanError as e:
        raise HTTPException(status_code=422, detail=str(e))

    result = detector.analyze_snippets(snippets)
    result["source"] = source
    result["total_ms"] = round((time.perf_counter() - start) * 1000)
    return result


if __name__ == "__main__":
    uvicorn.run(app, host=os.environ.get("HOST", "127.0.0.1"), port=int(os.environ.get("PORT", 8000)))
