"""
RAG service: Wraps ChromaDB vector search for retrieval-augmented generation.

Features:
  - Similarity scoring with configurable threshold
  - Context-window trimming for cheaper token usage
  - Metadata filtering support
"""

import os
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma

# Path to the persisted ChromaDB directory relative to project root
CHROMA_DB_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "chroma_db")

# Singleton vector store instance
_vectorstore: Chroma | None = None

# Similarity threshold — chunks scoring below this are discarded
SIMILARITY_THRESHOLD = 0.3

# Maximum total characters to return (context-window trimming)
MAX_CONTEXT_CHARS = 2000


def get_vectorstore() -> Chroma:
    """Get or initialise the ChromaDB vector store."""
    global _vectorstore
    if _vectorstore is None:
        embeddings = OpenAIEmbeddings()
        _vectorstore = Chroma(
            persist_directory=CHROMA_DB_DIR,
            embedding_function=embeddings
        )
    return _vectorstore


def retrieve_context(
    query: str,
    top_k: int = 6,
    life_area: str | None = None,
) -> list[str]:
    """
    Perform semantic search against the knowledge corpus.

    Args:
        query: The user question / search query.
        top_k: Number of candidate chunks to fetch before filtering.
        life_area: Optional metadata filter (e.g. 'career', 'love', 'spiritual',
                   'personality', 'planetary', 'nakshatra').

    Returns:
        List of relevant text passages that pass the similarity threshold,
        trimmed to fit within MAX_CONTEXT_CHARS.
    """
    vs = get_vectorstore()

    # Build optional metadata filter
    where_filter = None
    if life_area:
        where_filter = {"life_area": life_area}

    # Retrieve with scores for threshold filtering
    results_with_scores = vs.similarity_search_with_relevance_scores(
        query, k=top_k, filter=where_filter
    )

    # Filter by similarity threshold (higher score = more relevant)
    filtered = [
        (doc, score)
        for doc, score in results_with_scores
        if score >= SIMILARITY_THRESHOLD
    ]

    # Sort by score descending (best first)
    filtered.sort(key=lambda x: x[1], reverse=True)

    # Context-window trimming
    passages: list[str] = []
    total_chars = 0
    for doc, score in filtered:
        content = doc.page_content.strip()
        if total_chars + len(content) > MAX_CONTEXT_CHARS:
            # Include a truncated version if there's room
            remaining = MAX_CONTEXT_CHARS - total_chars
            if remaining > 100:
                passages.append(content[:remaining] + "…")
            break
        passages.append(content)
        total_chars += len(content)

    return passages
