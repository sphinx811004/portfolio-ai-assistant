"""
Public retrieval interface used by src/fingpt_signals and src/agent.
Keeps the rest of the codebase decoupled from FAISS/embedding details —
callers only ever see this module.

Alpha Vantage
      ↓
ingest_news.py
      ↓
news chunks
      ↓
document_store.py
      ↓
FAISS vector index
"""

from src.rag.document_store import DocumentStore
from src.rag.ingest_news import ingest_tickers

_backend = DocumentStore(name="news_index")
# _ implements that this variable is meant to be used by this module only and other modules should not depend on it



def refresh_index(tickers: list[str]) -> int:
    """
    Pulls fresh news for `tickers` from Alpha Vantage (respecting the on-disk
    cache in ingest_news) and adds any new chunks to the vector index.
    Returns the number of new chunks actually added.
    """
    chunks = ingest_tickers(tickers)
    return _backend.add_documents(chunks)


def retrieve_context(
        ticker: str, 
        query: str | None = None, 
        top_k: int = 5
    ) -> list[dict]:
    """
    Returns the top_k most relevant news chunks for a ticker. If no query is
    given, defaults to a generic "recent risk and outlook" query — useful when
    you just want the latest relevant context for a stock rather than
    answering something specific.
    """
    query = query or f"recent news, risks, and outlook for {ticker}"
    return _backend.search(
        query=query, 
        top_k=top_k, 
        ticker=ticker
    )


def format_context_for_prompt(chunks: list[dict]) -> str:
    """
    Formats retrieved chunks into a compact block suitable for dropping into
    a FinGPT / LLM prompt, with source attribution for traceability.
    """
    if not chunks:
        return "No relevant recent news found."

    lines = []
    for chunk in chunks:

        source = chunk.get("source", "unknown")
        published_at = chunk.get("published_at", "n/a")
        score = chunk.get("score", 0)
        lines.append(
            f"- [{source}, {published_at}] "
            f"{chunk['text']}"
            f"(relevance={score:.2f})"
        )

    return "\n".join(lines)


if __name__ == "__main__":
    added = refresh_index(["AAPL"])
    print(f"Added {added} new chunks.")

    results = retrieve_context("AAPL")

    print("\nRetrieved context:\n")
    print(format_context_for_prompt(results))
