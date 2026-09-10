"""
Public retrieval interface used by src/fingpt_signals and src/agent.
Keeps the rest of the codebase decoupled from FAISS/embedding details —
callers only ever see this module.
"""

from src.rag.document_store import DocumentStore
from src.rag.ingest_news import ingest_tickers

_store = DocumentStore(name="news_index")


def refresh_index(tickers: list[str]) -> int:
    """
    Pulls fresh news for `tickers` from Alpha Vantage (respecting the on-disk
    cache in ingest_news) and adds any new chunks to the vector index.
    Returns the number of new chunks actually added.
    """
    chunks = ingest_tickers(tickers)
    return _store.add_documents(chunks)


def retrieve_context(ticker: str, query: str = None, top_k: int = 5) -> list[dict]:
    """
    Returns the top_k most relevant news chunks for a ticker. If no query is
    given, defaults to a generic "recent risk and outlook" query — useful when
    you just want the latest relevant context for a stock rather than
    answering something specific.
    """
    query = query or f"recent news, risks, and outlook for {ticker}"
    return _store.search(query, top_k=top_k, ticker=ticker)


def format_context_for_prompt(chunks: list[dict]) -> str:
    """
    Formats retrieved chunks into a compact block suitable for dropping into
    a FinGPT / LLM prompt, with source attribution for traceability.
    """
    if not chunks:
        return "No relevant recent news found."

    lines = []
    for c in chunks:
        lines.append(
            f"- [{c.get('source', 'unknown')}, {c.get('published_at', 'n/a')}] "
            f"{c['text']} (relevance={c.get('score', 0):.2f})"
        )
    return "\n".join(lines)


if __name__ == "__main__":
    refresh_index(["AAPL"])
    results = retrieve_context("AAPL")
    print(format_context_for_prompt(results))
