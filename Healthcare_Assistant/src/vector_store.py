import numpy as np
from typing import List, Dict
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
from pathlib import Path
import json
import pickle


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
        # texts is a list of dicts: {'text': str, 'meta': {...}}
        self.texts: List[Dict] = []
        # plain_texts is a parallel list of strings used for embeddings/tfidf
        self.plain_texts: List[str] = []
        # TF-IDF fallback members (pure-Python)
        self.tfidf_vectors = None
        self.idf = None
        # Persistence paths (store FAISS index and metadata in data/faiss_index)
        self.index_dir = Path(__file__).resolve().parent.parent / "data" / "faiss_index"
        self.index_dir.mkdir(parents=True, exist_ok=True)
        self.faiss_index_path = self.index_dir / "index.faiss"
        self.texts_path = self.index_dir / "texts.json"
        self.tfidf_path = self.index_dir / "tfidf.pkl"

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

        # Try to load a persisted FAISS index if present. This avoids re-embedding
        # and re-building the index on every app start when the index was already
        # created previously.
        try:
            if self.faiss_index_path.exists() and self.texts_path.exists():
                try:
                    self.index = faiss.read_index(str(self.faiss_index_path))
                    with open(self.texts_path, 'r', encoding='utf-8') as f:
                        self.texts = json.load(f)
                    # Normalize older format (list of strings) to list of dicts
                    if self.texts and isinstance(self.texts[0], str):
                        self.texts = [{'text': t, 'meta': {}} for t in self.texts]
                    # Create parallel plain_texts list used for embeddings/tfidf
                    self.plain_texts = [t.get('text', '') if isinstance(t, dict) else str(t) for t in self.texts]
                    # If embeddings aren't available, keep mode as tfidf to avoid
                    # attempting to embed queries. If embeddings exist, keep openai mode.
                    if self.embeddings is None:
                        self.mode = 'tfidf'
                except Exception:
                    # If loading fails, remove partial files and continue
                    try:
                        self.faiss_index_path.unlink()
                    except Exception:
                        pass
        except Exception:
            pass

    def create_index(self, texts: List[Dict]):
        """Create an index from text chunks. `texts` should be a list of dicts
        with keys 'text' and optional 'meta'. Uses FAISS with OpenAI embeddings
        when available, otherwise fits a TF-IDF vectorizer as a local fallback.
        """
        # normalize incoming format: convert list[str] -> list[dict]
        if texts and isinstance(texts[0], str):
            texts = [{'text': t, 'meta': {}} for t in texts]

        self.texts = texts
        self.plain_texts = [t.get('text', '') if isinstance(t, dict) else str(t) for t in texts]

        if self.mode == "openai" and self.embeddings is not None:
            try:
                # If a persisted FAISS index exists and matches the number of
                # texts, reuse it instead of re-embedding.
                if self.faiss_index_path.exists() and self.texts_path.exists():
                    try:
                        with open(self.texts_path, 'r', encoding='utf-8') as f:
                            existing_texts = json.load(f)
                        # Normalize older format
                        if existing_texts and isinstance(existing_texts[0], str):
                            existing_texts = [{'text': t, 'meta': {}} for t in existing_texts]
                        if existing_texts == texts and self.index is not None:
                            # Index matches current texts; reuse it
                            return
                    except Exception:
                        # continue to rebuild index
                        pass

                # Create embeddings from plain texts
                embeddings = self.embeddings.embed_documents(self.plain_texts)
                embeddings_array = np.array(embeddings).astype("float32")

                # Create FAISS index
                dimension = embeddings_array.shape[1]
                self.index = faiss.IndexFlatL2(dimension)
                self.index.add(embeddings_array)

                # Persist index and texts for faster startup next time
                try:
                    faiss.write_index(self.index, str(self.faiss_index_path))
                    with open(self.texts_path, 'w', encoding='utf-8') as f:
                        json.dump(texts, f, ensure_ascii=False)
                except Exception:
                    # Non-fatal: if persistence fails, continue without it
                    traceback.print_exc()

                return
            except Exception:
                # If embeddings call failed (network), fall back to TF-IDF
                traceback.print_exc()
                self.mode = "tfidf"

        # TF-IDF fallback (pure-Python)
        if self.mode == "tfidf":
            # If a persisted TF-IDF exists and texts match, load it
            try:
                if self.tfidf_path.exists() and self.texts_path.exists():
                    with open(self.texts_path, 'r', encoding='utf-8') as f:
                        existing = json.load(f)
                    # Normalize older format
                    if existing and isinstance(existing[0], str):
                        existing = [{'text': t, 'meta': {}} for t in existing]
                    if existing == texts:
                        with open(self.tfidf_path, 'rb') as pf:
                            obj = pickle.load(pf)
                            self.tfidf_vectors = obj.get('tfidf_vectors')
                            self.idf = obj.get('idf')
                            return
            except Exception:
                pass

            # Fit TF-IDF on plain text strings
            self.tfidf_vectors, self.idf = _simple_tfidf_fit_transform(self.plain_texts)
            # Persist tfidf data for next runs
            try:
                with open(self.texts_path, 'w', encoding='utf-8') as f:
                    json.dump(texts, f, ensure_ascii=False)
                with open(self.tfidf_path, 'wb') as pf:
                    pickle.dump({'tfidf_vectors': self.tfidf_vectors, 'idf': self.idf}, pf)
            except Exception:
                pass

    def search(self, query: str, k: int = 5) -> List[dict]:
        """Search for relevant documents using the query.

        Returns a list of dicts with keys: 'text', 'doc_id', 'score', 'source'.
        'source' will be either 'faiss' or 'tfidf'. Scores are floats where
        higher usually indicates more relevant (for TF-IDF cosine similarity)
        and for FAISS the distance value is included (lower is better).
        """
        results = []
        if self.mode == "openai" and self.embeddings is not None and self.index is not None:
            try:
                query_embedding = self.embeddings.embed_query(query)
                query_embedding = np.array([query_embedding]).astype("float32")
                distances, indices = self.index.search(query_embedding, k)
                # distances is a 2D array of L2 distances; lower is better
                for dist, idx in zip(distances[0], indices[0]):
                    if idx < 0 or idx >= len(self.texts):
                        continue
                    item = self.texts[int(idx)]
                    meta = item.get('meta', {}) if isinstance(item, dict) else {}
                    results.append({
                        'text': item.get('text') if isinstance(item, dict) else item,
                        'doc_id': int(idx),
                        'score': float(dist),
                        'source': 'faiss',
                        'source_file': meta.get('source'),
                        'page': meta.get('page')
                    })
                return results
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
        for score, idx in sims[:k]:
            if score <= 0:
                continue
            item = self.texts[int(idx)]
            meta = item.get('meta', {}) if isinstance(item, dict) else {}
            results.append({
                'text': item.get('text') if isinstance(item, dict) else item,
                'doc_id': int(idx),
                'score': float(score),
                'source': 'tfidf',
                'source_file': meta.get('source'),
                'page': meta.get('page')
            })
        return results