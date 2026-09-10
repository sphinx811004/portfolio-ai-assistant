"""
Fetches news + sentiment articles for a list of tickers from Alpha Vantage's
NEWS_SENTIMENT endpoint, chunks each article, and hands the chunks off to the
document store (see document_store.py) for embedding + indexing.

Free-tier Alpha Vantage limits: 25 requests/day, 5 requests/minute.
This module caches raw responses to disk so re-runs don't burn your quota.
"""

import os
import json
import time
import hashlib
from pathlib import Path
from datetime import datetime, timezone

import requests
from dotenv import load_dotenv

load_dotenv()

ALPHA_VANTAGE_API_KEY = os.getenv("ALPHA_VANTAGE_API_KEY")
BASE_URL = "https://www.alphavantage.co/query"
CACHE_DIR = Path("data/cache/news")
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# Be conservative: free tier allows 5/min. Space calls out.
SECONDS_BETWEEN_CALLS = 15


def _cache_path(ticker: str) -> Path:
    key = hashlib.md5(ticker.encode()).hexdigest()[:10]
    return CACHE_DIR / f"{ticker}_{key}.json"


def fetch_news_sentiment(ticker: str, limit: int = 50, max_age_hours: int = 12) -> list[dict]:
    """
    Returns Alpha Vantage's raw article feed for a ticker, using an on-disk
    cache so we don't re-spend API quota within `max_age_hours`.
    """
    cache_file = _cache_path(ticker)

    if cache_file.exists():
        age_hours = (time.time() - cache_file.stat().st_mtime) / 3600
        if age_hours < max_age_hours:
            with open(cache_file) as f:
                return json.load(f)["feed"]

    if not ALPHA_VANTAGE_API_KEY:
        raise RuntimeError("ALPHA_VANTAGE_API_KEY is not set — check your .env file.")

    params = {
        "function": "NEWS_SENTIMENT",
        "tickers": ticker,
        "limit": limit,
        "apikey": ALPHA_VANTAGE_API_KEY,
    }
    resp = requests.get(BASE_URL, params=params, timeout=15)
    resp.raise_for_status()
    data = resp.json()

    if "feed" not in data:
        # Common cases: rate limit hit, or invalid ticker/key.
        raise RuntimeError(f"Unexpected Alpha Vantage response for {ticker}: {data}")

    with open(cache_file, "w") as f:
        json.dump(data, f)

    return data["feed"]


def articles_to_documents(ticker: str, feed: list[dict]) -> list[dict]:
    """
    Converts raw Alpha Vantage feed items into normalized documents ready for
    chunking + embedding. Each doc keeps metadata needed for citation/filtering
    (source, publish time, per-ticker sentiment score).
    """
    docs = []
    for item in feed:
        # ticker_sentiment is a list of per-ticker relevance/sentiment scores
        ticker_score = next(
            (t for t in item.get("ticker_sentiment", []) if t.get("ticker") == ticker),
            None,
        )
        text = f"{item.get('title', '')}\n\n{item.get('summary', '')}".strip()
        if not text:
            continue

        docs.append({
            "id": hashlib.md5(item.get("url", text).encode()).hexdigest(),
            "ticker": ticker,
            "text": text,
            "source": item.get("source"),
            "url": item.get("url"),
            "published_at": item.get("time_published"),
            "overall_sentiment_score": item.get("overall_sentiment_score"),
            "relevance_score": float(ticker_score["relevance_score"]) if ticker_score else None,
            "ticker_sentiment_score": float(ticker_score["ticker_sentiment_score"]) if ticker_score else None,
            "ingested_at": datetime.now(timezone.utc).isoformat(),
        })
    return docs


def chunk_document(doc: dict, max_words: int = 200, overlap: int = 40) -> list[dict]:
    """
    Splits a document's text into overlapping word-chunks. Most Alpha Vantage
    articles are short (title + summary), so this is usually a single chunk —
    but the function scales cleanly if you later ingest longer text
    (e.g. EARNINGS_CALL_TRANSCRIPT).
    """
    words = doc["text"].split()
    if len(words) <= max_words:
        return [{**doc, "chunk_id": f"{doc['id']}_0"}]

    chunks = []
    start = 0
    idx = 0
    while start < len(words):
        chunk_words = words[start:start + max_words]
        chunks.append({
            **doc,
            "chunk_id": f"{doc['id']}_{idx}",
            "text": " ".join(chunk_words),
        })
        start += max_words - overlap
        idx += 1
    return chunks


def ingest_tickers(tickers: list[str]) -> list[dict]:
    """
    End-to-end ingestion for a list of tickers: fetch -> normalize -> chunk.
    Returns a flat list of chunk-level documents ready for the document store.
    """
    all_chunks = []
    for i, ticker in enumerate(tickers):
        feed = fetch_news_sentiment(ticker)
        docs = articles_to_documents(ticker, feed)
        for doc in docs:
            all_chunks.extend(chunk_document(doc))

        # Only sleep between *live* calls, not cache hits — fetch_news_sentiment
        # already short-circuits on cache, so this just protects fresh fetches.
        if i < len(tickers) - 1:
            time.sleep(SECONDS_BETWEEN_CALLS)

    return all_chunks


if __name__ == "__main__":
    chunks = ingest_tickers(["AAPL", "MSFT"])
    print(f"Ingested {len(chunks)} chunks")
