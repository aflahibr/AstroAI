"""
Knowledge corpus ingestion script.

Loads all data files from /data, creates chunked documents with metadata
tags (zodiac, life_area, planetary), embeds them, and stores in ChromaDB.
"""

import os
import json
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from dotenv import load_dotenv

load_dotenv()

# Configuration
CHROMA_DB_DIR = "chroma_db"
DATA_DIR = "data"


def load_text_data(filepath: str, life_area: str) -> list[Document]:
    """Load a text file, split into chunks, and tag with metadata."""
    print(f"Loading {filepath}...")
    loader = TextLoader(filepath, encoding="utf-8")
    docs = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=200)
    chunks = text_splitter.split_documents(docs)
    for chunk in chunks:
        chunk.metadata["life_area"] = life_area
        chunk.metadata["source_file"] = os.path.basename(filepath)
    return chunks


def load_zodiac_traits(filepath: str) -> list[Document]:
    """Load zodiac_traits.json — nested {sign: {personality, strengths, challenges}}."""
    print(f"Loading {filepath}...")
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)

    docs = []
    for sign, traits in data.items():
        content = (
            f"{sign} Zodiac Traits\n"
            f"Personality: {traits['personality']}\n"
            f"Strengths: {traits['strengths']}\n"
            f"Challenges: {traits['challenges']}"
        )
        docs.append(
            Document(
                page_content=content,
                metadata={
                    "zodiac": sign,
                    "life_area": "personality",
                    "source_file": os.path.basename(filepath),
                },
            )
        )
    return docs


def load_planetary_impacts(filepath: str) -> list[Document]:
    """Load planetary_impacts.json — nested {planet: {description, nature, influence}}."""
    print(f"Loading {filepath}...")
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)

    docs = []
    for planet, info in data.items():
        content = (
            f"{planet}\n"
            f"{info['description']}\n"
            f"Nature: {info['nature']} | Influence: {info['influence']}"
        )
        docs.append(
            Document(
                page_content=content,
                metadata={
                    "planetary": planet,
                    "nature": info["nature"],
                    "influence": info["influence"],
                    "life_area": "planetary",
                    "source_file": os.path.basename(filepath),
                },
            )
        )
    return docs


def load_flat_json(filepath: str, life_area: str) -> list[Document]:
    """Load a flat JSON file {key: value} into tagged documents."""
    print(f"Loading {filepath}...")
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)

    docs = []
    for key, value in data.items():
        docs.append(
            Document(
                page_content=f"{key}: {value}",
                metadata={
                    "life_area": life_area,
                    "source_file": os.path.basename(filepath),
                },
            )
        )
    return docs


def ingest():
    print("Starting ingestion process...")

    all_docs: list[Document] = []

    # --- Text files (career, love, spiritual guidance) ---
    for filename, area in [
        ("career_guidance.txt", "career"),
        ("love_guidance.txt", "love"),
        ("spiritual_guidance.txt", "spiritual"),
    ]:
        path = os.path.join(DATA_DIR, filename)
        if os.path.exists(path):
            all_docs.extend(load_text_data(path, life_area=area))

    # --- Structured JSON files ---
    zodiac_path = os.path.join(DATA_DIR, "zodiac_traits.json")
    if os.path.exists(zodiac_path):
        all_docs.extend(load_zodiac_traits(zodiac_path))

    planetary_path = os.path.join(DATA_DIR, "planetary_impacts.json")
    if os.path.exists(planetary_path):
        all_docs.extend(load_planetary_impacts(planetary_path))

    # --- Optional: Nakshatra mapping ---
    nakshatra_path = os.path.join(DATA_DIR, "nakshatra_mapping.json")
    if os.path.exists(nakshatra_path):
        all_docs.extend(load_flat_json(nakshatra_path, life_area="nakshatra"))

    print(f"Total documents to embed: {len(all_docs)}")

    # Embed and store
    embeddings = OpenAIEmbeddings()
    print(f"Creating ChromaDB at {CHROMA_DB_DIR}...")

    db = Chroma.from_documents(
        documents=all_docs,
        embedding=embeddings,
        persist_directory=CHROMA_DB_DIR,
    )

    print("Ingestion completed successfully.")


if __name__ == "__main__":
    ingest()
