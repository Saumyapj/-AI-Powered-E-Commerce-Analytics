"""
build_index.py
===============
One-time (or re-run-when-data-changes) script that embeds the business
knowledge base (models/artifacts/knowledge/*.txt) and upserts it into a
Pinecone index, so the Streamlit app's "Ask Your Data" tab has something
to retrieve from.

This is intentionally separate from rag_engine.py: re-embedding and
re-uploading every file on every Streamlit app start would be slow and
wasteful. Run this script whenever the knowledge/*.txt files change,
then just run the Streamlit app as usual.

Usage:
    export GOOGLE_API_KEY=...        # not needed for this script, but the
                                      # app will need it later
    export PINECONE_API_KEY=...
    python build_index.py
"""

import os

# Must be set BEFORE numpy/torch/sentence-transformers are imported anywhere
# below (they read these at import time). Without this, OpenBLAS tries to
# allocate a full thread pool sized to the host's CPU count, which can
# exhaust memory on small/shared containers (e.g. Streamlit Cloud free tier)
# and crash with "Memory allocation still failed after 10 retries".
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

from pathlib import Path

from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec

def _find_project_root(start: Path) -> Path:
    """Walk upward from `start` looking for a project-root marker
    (.env file, or a models/ directory). Falls back to `start` itself
    if nothing is found, so behavior degrades gracefully."""
    current = start.resolve()
    for candidate in [current, *current.parents]:
        if (candidate / ".env").exists() or (candidate / "models").is_dir():
            return candidate
    return start.resolve()


ROOT = _find_project_root(Path(__file__).resolve().parent)
KNOWLEDGE_DIR = ROOT / "models" / "artifacts" / "knowledge"

try:
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")  # no-op if the file doesn't exist
except ImportError:
    pass  # python-dotenv not installed; rely on real env vars instead

PINECONE_INDEX_NAME = os.environ.get("PINECONE_INDEX_NAME", "chatbot")
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIM = 384  # dimension of all-MiniLM-L6-v2 embeddings

CHUNK_SIZE = 500
CHUNK_OVERLAP = 20


def load_txt_files(data_dir: Path):
    loader = DirectoryLoader(str(data_dir), glob="*.txt", loader_cls=TextLoader)
    return loader.load()


def minimal_docs(docs):
    """Keep only page_content + source, dropping other loader metadata."""
    from langchain_core.documents import Document
    return [Document(page_content=d.page_content, metadata={"source": d.metadata.get("source")}) for d in docs]


def split_docs(docs):
    splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    return splitter.split_documents(docs)


def main():
    pinecone_api_key = os.environ.get("PINECONE_API_KEY")
    if not pinecone_api_key:
        raise SystemExit(
            "PINECONE_API_KEY is not set. Get one at https://app.pinecone.io "
            "and export it before running this script."
        )

    if not KNOWLEDGE_DIR.exists():
        raise SystemExit(f"Knowledge directory not found: {KNOWLEDGE_DIR}")

    print(f"Loading .txt files from {KNOWLEDGE_DIR} ...")
    raw_docs = load_txt_files(KNOWLEDGE_DIR)
    print(f"  loaded {len(raw_docs)} file(s)")

    docs = minimal_docs(raw_docs)
    chunks = split_docs(docs)
    print(f"  split into {len(chunks)} chunk(s)")

    print(f"Loading embedding model '{EMBEDDING_MODEL_NAME}' (first run downloads it) ...")
    embedding = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"batch_size": 16},
    )

    pc = Pinecone(api_key=pinecone_api_key)
    existing = [i["name"] for i in pc.list_indexes()]
    if PINECONE_INDEX_NAME not in existing:
        print(f"Creating Pinecone index '{PINECONE_INDEX_NAME}' (dim={EMBEDDING_DIM}, metric=cosine) ...")
        pc.create_index(
            name=PINECONE_INDEX_NAME,
            dimension=EMBEDDING_DIM,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        )
    else:
        print(f"Pinecone index '{PINECONE_INDEX_NAME}' already exists — will upsert into it.")

    print(f"Embedding and upserting {len(chunks)} chunk(s) into Pinecone ...")
    PineconeVectorStore.from_documents(
        documents=chunks,
        embedding=embedding,
        index_name=PINECONE_INDEX_NAME,
    )

    stats = pc.Index(PINECONE_INDEX_NAME).describe_index_stats()
    print("Done. Index stats:", stats)


if __name__ == "__main__":
    main()