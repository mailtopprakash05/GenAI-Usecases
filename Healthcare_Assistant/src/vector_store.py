import numpy as np
from typing import List
import faiss
import traceback
# Try to import a LangChain OpenAI embeddings wrapper from a couple of
# possible package locations. If those imports fail, we'll fall back to a
# tiny internal wrapper that calls the `openai` package directly.
EmbeddingClass = None
try:
    from langchain_community.embeddings import OpenAIEmbeddings as LangChainOpenAIEmbeddings
    EmbeddingClass = LangChainOpenAIEmbeddings
except Exception:
    try:
        from langchain.embeddings import OpenAIEmbeddings as LangChainOpenAIEmbeddings2
        EmbeddingClass = LangChainOpenAIEmbeddings2
    except Exception:
        EmbeddingClass = None

try:
    import openai
except Exception:
    openai = None
from collections import Counter
import math


def _simple_tfidf_fit_transform(docs: List[str]):
    # Build vocabulary and compute idf
    tokenized = [doc.lower().split() for doc in docs]
    vocab = {}
    df = Counter()
    for tokens in tokenized:
        unique = set(tokens)
        for t in unique:
            df[t] += 1

    idf = {t: math.log((1 + len(docs)) / (1 + df_t)) + 1 for t, df_t in df.items()}

    # Compute tf-idf vectors as dicts
    vectors = []
    for tokens in tokenized:
        tf = Counter(tokens)
        vec = {t: (tf[t] * idf.get(t, 0.0)) for t in tf.keys()}
        # normalize
        norm = math.sqrt(sum(v * v for v in vec.values())) or 1.0
        for k in list(vec.keys()):
            vec[k] = vec[k] / norm
        vectors.append(vec)

    return vectors, idf


def _simple_cosine_similarity(vec_q, vec_d):
    # both are dicts
    dot = 0.0
    for k, v in vec_q.items():
        dot += v * vec_d.get(k, 0.0)
    # norms are assumed to be 1.0 (we normalized during fit)
    return dot


# Minimal wrapper that uses the official `openai` package to produce
# embeddings when a LangChain wrapper is not available. It exposes the
# small subset of methods used by the rest of this module: `embed_documents`
# and `embed_query`.
class _SimpleOpenAIEmbeddings:
    def __init__(self, openai_api_key: str | None = None, model: str = "text-embedding-3-small"):
        if openai is None:
            raise RuntimeError("openai package is required for _SimpleOpenAIEmbeddings")
        self.model = model
        if openai_api_key:
            openai.api_key = openai_api_key

    def embed_documents(self, texts: List[str]):
        # openai.Embeddings.create accepts a list of inputs
        resp = openai.Embeddings.create(model=self.model, input=texts)
        return [d["embedding"] for d in resp["data"]]

    def embed_query(self, text: str):
        resp = openai.Embeddings.create(model=self.model, input=text)
        return resp["data"][0]["embedding"]


class VectorStore:
    """Vector store with OpenAI+FAISS primary path and TF-IDF fallback.

    This makes the RAG retrieval resilient when the environment has no network
    access (or DNS issues) preventing OpenAI/tiktoken downloads. In that case
    the store will fall back to a local TF-IDF index.
    """

    def __init__(self, api_key: str | None = None):
        self.api_key = api_key
        self.mode = "openai"
        self.index = None
        self.texts: List[str] = []
        # TF-IDF fallback members (pure-Python)
        self.tfidf_vectors = None
        self.idf = None

        # Try to initialize OpenAI embeddings. If this fails (network/DNS),
        # we'll switch to TF-IDF mode.
        # Prefer an available LangChain wrapper if one was imported earlier
        if EmbeddingClass is not None:
            try:
                self.embeddings = EmbeddingClass(openai_api_key=api_key)
            except Exception:
                traceback.print_exc()
                self.embeddings = None
                self.mode = "tfidf"
        else:
            # Fall back to the tiny openai-based wrapper (if openai is installed)
            try:
                self.embeddings = _SimpleOpenAIEmbeddings(openai_api_key=api_key)
            except Exception:
                traceback.print_exc()
                self.embeddings = None
                self.mode = "tfidf"

    def create_index(self, texts: List[str]):
        """Create an index from text chunks. Uses FAISS with OpenAI embeddings
        when available, otherwise fits a TF-IDF vectorizer as a local fallback.
        """
        self.texts = texts

        if self.mode == "openai" and self.embeddings is not None:
            try:
                embeddings = self.embeddings.embed_documents(texts)
                embeddings_array = np.array(embeddings).astype("float32")

                # Create FAISS index
                dimension = embeddings_array.shape[1]
                self.index = faiss.IndexFlatL2(dimension)
                self.index.add(embeddings_array)
                return
            except Exception:
                # If embeddings call failed (network), fall back to TF-IDF
                traceback.print_exc()
                self.mode = "tfidf"

        # TF-IDF fallback (pure-Python)
        if self.mode == "tfidf":
            self.tfidf_vectors, self.idf = _simple_tfidf_fit_transform(texts)

    def search(self, query: str, k: int = 5) -> List[str]:
        """Search for relevant documents using the query.

        Returns the top-k text chunks (or fewer if not available).
        """
        if self.mode == "openai" and self.embeddings is not None and self.index is not None:
            try:
                query_embedding = self.embeddings.embed_query(query)
                query_embedding = np.array([query_embedding]).astype("float32")
                distances, indices = self.index.search(query_embedding, k)
                return [self.texts[i] for i in indices[0] if i < len(self.texts)]
            except Exception:
                # If embeddings fail at query time, switch to TF-IDF fallback
                traceback.print_exc()
                self.mode = "tfidf"

        # TF-IDF fallback search (pure-Python)
        if not getattr(self, "tfidf_vectors", None):
            return []

        # build query vector using idf
        tokens = query.lower().split()
        tf = Counter(tokens)
        q_vec = {}
        for t, cnt in tf.items():
            if t in self.idf:
                q_vec[t] = cnt * self.idf.get(t, 0.0)
        # normalize
        norm = math.sqrt(sum(v * v for v in q_vec.values())) or 1.0
        for k2 in list(q_vec.keys()):
            q_vec[k2] = q_vec[k2] / norm

        sims = [(_simple_cosine_similarity(q_vec, d), idx) for idx, d in enumerate(self.tfidf_vectors)]
        sims.sort(key=lambda x: x[0], reverse=True)
        results = [self.texts[idx] for score, idx in sims[:k] if score > 0]
        return results