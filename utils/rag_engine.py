"""
utils/rag_engine.py
====================
Core RAG (Retrieval-Augmented Generation) engine that powers the
"Ask Your Data" tab of the E-Commerce Analytics Platform
(frontend/streamlit_app.py).

Design (matches the Chatbot_E-Commerce.ipynb prototype):
- RETRIEVAL is a Pinecone vector index over chunks of the knowledge/*.txt
  files, embedded with a local HuggingFace sentence-transformers model
  (no embedding API calls — only Pinecone needs a network round trip).
- GENERATION is done with Google's Gemini API (via langchain-google-genai),
  given ONLY the retrieved chunks as context and instructed to answer
  strictly from them. This keeps every answer grounded in the actual
  business dataset analysis rather than the model's general knowledge.

Requires two credentials, set as environment variables (or Streamlit
secrets):
    GOOGLE_API_KEY   — https://aistudio.google.com/apikey
    PINECONE_API_KEY — https://app.pinecone.io  (API Keys page)

The Pinecone index itself is NOT built automatically by this module (that
would mean re-embedding and re-uploading every file on every app start).
Run `build_index.py` once beforehand to create and populate it. This
module only ever *reads* from an existing index.

The knowledge/ directory (at the project ROOT, alongside models/ and
frontend/) is expected to contain the following files, produced from the
Olist e-commerce capstone analysis:
    dataset_overview.txt, sales_kpis.txt, product_analysis.txt,
    customer_analysis.txt, delivery_analysis.txt, sentiment_analysis.txt,
    forecasting_results.txt, recommendation_results.txt,
    business_insights.txt
"""

import os

# Must be set BEFORE numpy/torch/sentence-transformers are imported anywhere
# below (they read these at import time). Prevents OpenBLAS from trying to
# allocate a full CPU-sized thread pool, which can OOM on small/shared
# containers. See build_index.py for the matching guard.
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

from langchain_pinecone import PineconeVectorStore
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage
from pinecone import Pinecone

def _find_project_root(start: Path) -> Path:
    """Walk upward from `start` looking for a project-root marker
    (.env file, or a models/ directory). Falls back one level up from
    `start` if nothing is found (this file's historical default)."""
    current = start.resolve()
    for candidate in [current, *current.parents]:
        if (candidate / ".env").exists() or (candidate / "models").is_dir():
            return candidate
    return start.resolve().parent


# utils/rag_engine.py -> project ROOT is normally one level up from utils/,
# but we auto-detect in case the folder layout changes.
ROOT = _find_project_root(Path(__file__).resolve().parent)
KNOWLEDGE_DIR = ROOT / "models" / "artifacts" / "knowledge"

try:
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")  # no-op if the file doesn't exist
except ImportError:
    pass  # python-dotenv not installed; rely on real env vars / st.secrets bridge instead

# Must match the index build_index.py created.
PINECONE_INDEX_NAME = os.environ.get("PINECONE_INDEX_NAME", "chatbot")

# Must match the model build_index.py embedded with (same dimension: 384).
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

DEFAULT_MODEL = "gemini-3.6-flash"


# ---------------------------------------------------------------------
# Shared retrieval-result type
# ---------------------------------------------------------------------

@dataclass
class Chunk:
    text: str
    source: str      # filename, e.g. "sales_kpis.txt"
    section: str      # readable label derived from the filename


def _label_from_source(source: str) -> str:
    return Path(source).stem.replace("_", " ").title()


# ---------------------------------------------------------------------
# Embeddings / Pinecone connection (built lazily, cached at module level)
# ---------------------------------------------------------------------

_embedding = None


def _get_embedding() -> HuggingFaceEmbeddings:
    global _embedding
    if _embedding is None:
        _embedding = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL_NAME,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"batch_size": 16},
        )
    return _embedding


def _require_env(var_name: str, help_url: str) -> str:
    value = os.environ.get(var_name)
    if not value:
        raise RuntimeError(
            f"No {var_name} found. Set it as an environment variable "
            f"(or Streamlit secret). Get one at {help_url}."
        )
    return value


class RAGIndex:
    """Thin wrapper around an existing Pinecone index + HuggingFace
    embeddings, exposing a `.retrieve()` call similar to a local index."""

    def __init__(self, index_name: str = PINECONE_INDEX_NAME):
        pinecone_api_key = _require_env("PINECONE_API_KEY", "https://app.pinecone.io")
        # GOOGLE_API_KEY isn't needed for retrieval, but fail fast here too
        # so the whole chat tab degrades gracefully in one place.
        _require_env("GOOGLE_API_KEY", "https://aistudio.google.com/apikey")

        pc = Pinecone(api_key=pinecone_api_key)
        existing = [i["name"] for i in pc.list_indexes()]
        if index_name not in existing:
            raise RuntimeError(
                f"Pinecone index '{index_name}' does not exist yet. "
                f"Run `python build_index.py` once to create and populate it "
                f"from the files in {KNOWLEDGE_DIR}."
            )

        self.index_name = index_name
        self._pc = pc
        self.vectorstore = PineconeVectorStore.from_existing_index(
            index_name=index_name, embedding=_get_embedding()
        )

    def retrieve(self, query: str, top_k: int = 5) -> List[Tuple[Chunk, float]]:
        """Return up to top_k (chunk, similarity_score) pairs, best first."""
        results = self.vectorstore.similarity_search_with_score(query, k=top_k)
        out = []
        for doc, score in results:
            source = os.path.basename(doc.metadata.get("source", "unknown.txt"))
            out.append(
                (Chunk(text=doc.page_content, source=source, section=_label_from_source(source)), float(score))
            )
        return out

    def stats(self) -> dict:
        desc = self._pc.Index(self.index_name).describe_index_stats()
        return {"index_name": self.index_name, "total_vectors": desc.get("total_vector_count", 0)}


# ---------------------------------------------------------------------
# Generation (Gemini call, grounded in retrieved chunks)
# ---------------------------------------------------------------------

SYSTEM_PROMPT = """You are an AI business analyst for an e-commerce company. \
You answer questions from business owners and managers using ONLY the context \
excerpts provided to you below, which come from real analysis of the \
company's own order, customer, delivery, review, and forecasting data.

Rules:
1. Base every claim strictly on the provided context. Do not use outside \
knowledge about e-commerce in general, and do not invent numbers.
2. If the context does not contain enough information to answer the \
question, say so plainly and suggest what analysis would be needed \
instead of guessing.
3. Be concise and direct, the way you'd brief a busy business owner: \
lead with the answer, then the supporting numbers.
4. When useful, translate raw statistics into a plain-English business \
takeaway or recommendation, but clearly distinguish "what the data shows" \
from "what we'd suggest doing about it."
5. Prefer concrete numbers from the context over vague language.
6. Do not mention "chunks," "context blocks," or retrieval mechanics in \
your answer - just answer naturally as an analyst would.
"""


def _build_user_prompt(question: str, retrieved: List[Tuple[Chunk, float]]) -> str:
    context_blocks = []
    for chunk, _score in retrieved:
        context_blocks.append(f"[Source: {chunk.source} | {chunk.section}]\n{chunk.text}")
    context = "\n\n---\n\n".join(context_blocks)
    return (
        f"CONTEXT FROM THE BUSINESS KNOWLEDGE BASE:\n\n{context}\n\n"
        f"---\n\nQUESTION: {question}\n\n"
        "Answer the question using only the context above."
    )


class NoRetrievedContext(Exception):
    pass


def _extract_text(content) -> str:
    """Normalize a LangChain message's .content into a plain string.
    Newer langchain-google-genai versions may return a list of content
    parts (e.g. [{"type": "text", "text": "..."}]) instead of a plain
    string, depending on the response shape Gemini returns."""
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                parts.append(item.get("text", "") or item.get("content", ""))
        return "".join(parts).strip()
    return str(content or "").strip()


def _generate_with_gemini(model: str, user_prompt: str) -> str:
    """Call the Gemini API via langchain-google-genai. Requires GOOGLE_API_KEY."""
    api_key = _require_env("GOOGLE_API_KEY", "https://aistudio.google.com/apikey")
    chat_model = ChatGoogleGenerativeAI(model=model, google_api_key=api_key)
    response = chat_model.invoke(
        [SystemMessage(content=SYSTEM_PROMPT), HumanMessage(content=user_prompt)]
    )
    return _extract_text(response.content)


def answer_question(
    question: str,
    index: RAGIndex,
    top_k: int = 5,
    model: str = DEFAULT_MODEL,
) -> Tuple[str, List[str], List[Tuple[Chunk, float]]]:
    """
    Retrieve relevant chunks from Pinecone and ask Gemini to answer
    grounded in them.

    Returns: (answer_text, sorted_unique_source_filenames, retrieved_chunks)

    Raises NoRetrievedContext if nothing in the knowledge base is
    relevant to the question (caller should show a friendly "not covered
    by this dataset" message rather than calling the LLM with empty
    context).
    """
    retrieved = index.retrieve(question, top_k=top_k)
    if not retrieved:
        raise NoRetrievedContext(
            "No relevant information was found in the business knowledge base "
            "for this question."
        )

    sources = sorted(set(c.source for c, _ in retrieved))
    user_prompt = _build_user_prompt(question, retrieved)
    answer_text = _generate_with_gemini(model, user_prompt)

    return answer_text, sources, retrieved


def extractive_fallback(question: str, index: RAGIndex, top_k: int = 3) -> Tuple[str, List[str], List[Tuple[Chunk, float]]]:
    """
    A no-LLM fallback: just surface the most relevant raw excerpts.
    Used when GOOGLE_API_KEY isn't configured, so the retrieval half of
    the system is still fully testable and useful on its own.
    """
    retrieved = index.retrieve(question, top_k=top_k)
    if not retrieved:
        return (
            "I couldn't find anything in the business knowledge base related to that "
            "question. Try rephrasing, or ask about sales, customers, delivery, "
            "sentiment, forecasting, or recommendations.",
            [],
            [],
        )
    parts = []
    for chunk, score in retrieved:
        parts.append(f"**From {chunk.source} — {chunk.section}** (relevance {score:.2f}):\n{chunk.text}")
    sources = sorted(set(c.source for c, _ in retrieved))
    text = (
        "*(No LLM available — showing the most relevant raw excerpts instead of "
        "a generated answer. Set GOOGLE_API_KEY to enable full generated answers.)*\n\n"
        + "\n\n---\n\n".join(parts)
    )
    return text, sources, retrieved


if __name__ == "__main__":
    # Quick manual smoke test of retrieval + generation (requires
    # PINECONE_API_KEY, GOOGLE_API_KEY, and a populated index — run
    # build_index.py first).
    idx = RAGIndex()
    print("Index stats:", idx.stats())
    for q in [
        "What is our total revenue and average order value?",
        "Why do late deliveries hurt customer satisfaction?",
    ]:
        print("\n" + "=" * 70)
        print("Q:", q)
        answer, sources, retrieved = answer_question(q, idx, top_k=3)
        print("A:", answer)
        print("Sources:", sources)