"""
RAG service: Wraps ChromaDB vector search for retrieval-augmented generation.
"""

import os
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma

# Path to the persisted ChromaDB directory relative to project root
CHROMA_DB_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "chroma_db")

# Singleton vector store instance
_vectorstore: Chroma | None = None


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


def retrieve_context(query: str, top_k: int = 4) -> list[str]:
    """
    Perform semantic search against the knowledge corpus.
    
    Args:
        query: The user question / search query.
        top_k: Number of top chunks to return.
    
    Returns:
        List of relevant text passages from the knowledge base.
    """
    vs = get_vectorstore()
    results = vs.similarity_search(query, k=top_k)
    return [doc.page_content for doc in results]
