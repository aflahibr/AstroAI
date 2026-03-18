import os
import json
from langchain_community.document_loaders import TextLoader, JSONLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from dotenv import load_dotenv

load_dotenv()


# Configuration
CHROMA_DB_DIR = "chroma_db"
DATA_DIR = "data"


def load_text_data(filepath: str):
    print(f"Loading {filepath}...")
    loader = TextLoader(filepath)
    docs = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    return text_splitter.split_documents(docs)


def load_json_data(filepath: str, jq_schema: str):
    print(f"Loading {filepath}...")
    # Using a simple JSON loading and text generation since JSONLoader can be tricky with structures
    docs = []
    with open(filepath, "r") as f:
        data = json.load(f)
        for key, value in data.items():
            # Creating descriptive mock documents combining keys and values
            content = f"{key}: {value}"
            docs.append(content)

    from langchain_core.documents import Document

    return [Document(page_content=d) for d in docs]


def ingest():
    print("Starting ingestion process...")

    # 1. Load documents
    all_docs = []

    # Text data
    text_docs = load_text_data(os.path.join(DATA_DIR, "vedic_astrology.txt"))
    all_docs.extend(text_docs)

    # JSON data
    planetary_docs = load_json_data(
        os.path.join(DATA_DIR, "planetary_traits.json"), ".[]"
    )
    zodiac_docs = load_json_data(
        os.path.join(DATA_DIR, "zodiac_personality.json"), ".[]"
    )

    all_docs.extend(planetary_docs)
    all_docs.extend(zodiac_docs)

    print(f"Total documents to embed: {len(all_docs)}")

    # 2. Embed and Store
    embeddings = OpenAIEmbeddings()
    print(f"Creating ChromaDB at {CHROMA_DB_DIR}...")

    # Initialize chroma db
    db = Chroma.from_documents(
        documents=all_docs, embedding=embeddings, persist_directory=CHROMA_DB_DIR
    )

    print("Ingestion completed successfully.")


if __name__ == "__main__":
    ingest()
