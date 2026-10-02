"""
Embeddings + a simple local vector store.

- Embeddings: a multilingual sentence-transformers model (default
  intfloat/multilingual-e5-small) running locally on CPU.
- Vector store: a normalized numpy matrix (embeddings.npy) plus the chunk
  texts/metadata (chunks.json) in data/index/. Similarity search is a single
  matrix-vector product (cosine similarity), which is fast enough for tens of
  thousands of chunks and needs no extra infrastructure.

The index records a fingerprint of the source data so a stale index (database
changed since the last build) can be detected and rebuilt.
"""

import hashlib
import json
import logging
import os
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import numpy as np

from config import EMBEDDING_MODEL, INDEX_DIR, TOP_K

logger = logging.getLogger(__name__)

EMBEDDINGS_FILE = "embeddings.npy"
CHUNKS_FILE = "chunks.json"
META_FILE = "meta.json"


class Embedder:
    """Wraps a sentence-transformers model. Loaded lazily on first use."""

    def __init__(self, model_name: str = EMBEDDING_MODEL):
        self.model_name = model_name
        self._model = None
        # e5 models are trained with these prefixes; other models don't need them
        self._is_e5 = "e5" in model_name.lower()

    def _load(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            logger.info(f"Loading embedding model {self.model_name} ...")
            try:
                # Use the cached copy without contacting the Hugging Face Hub, so
                # the app works offline and never hangs on a slow network.
                self._model = SentenceTransformer(self.model_name, local_files_only=True)
            except Exception:
                logger.info("Model not in local cache; downloading (first run only) ...")
                self._model = SentenceTransformer(self.model_name)
        return self._model

    def _encode(self, texts: List[str]) -> np.ndarray:
        vectors = self._load().encode(
            texts,
            batch_size=32,
            normalize_embeddings=True,
            show_progress_bar=len(texts) > 64,
            convert_to_numpy=True,
        )
        return np.asarray(vectors, dtype=np.float32)

    def encode_passages(self, texts: List[str]) -> np.ndarray:
        prefix = "passage: " if self._is_e5 else ""
        return self._encode([prefix + t for t in texts])

    def encode_query(self, text: str) -> np.ndarray:
        prefix = "query: " if self._is_e5 else ""
        return self._encode([prefix + text])[0]


def data_fingerprint(records: List[Dict[str, Any]]) -> str:
    """Hash of (serial, content) pairs; changes whenever source content changes."""
    h = hashlib.sha256()
    for r in sorted(records, key=lambda r: r.get("serial") or ""):
        h.update((r.get("serial") or "").encode("utf-8"))
        h.update(b"\0")
        h.update((r.get("content") or "").encode("utf-8"))
        h.update(b"\0")
    return h.hexdigest()


class VectorStore:
    """In-memory cosine-similarity index persisted as .npy + .json files."""

    def __init__(self, embeddings: np.ndarray, chunks: List[Dict[str, Any]], meta: Dict[str, Any]):
        if len(embeddings) != len(chunks):
            raise ValueError("Embeddings and chunks are out of sync")
        self.embeddings = embeddings
        self.chunks = chunks
        self.meta = meta

    def __len__(self) -> int:
        return len(self.chunks)

    @classmethod
    def build(
        cls,
        chunks: List[Dict[str, Any]],
        embedder: Embedder,
        fingerprint: str = "",
    ) -> "VectorStore":
        if chunks:
            embeddings = embedder.encode_passages([c["embed_text"] for c in chunks])
        else:
            embeddings = np.zeros((0, 0), dtype=np.float32)
        meta = {
            "embedding_model": embedder.model_name,
            "num_chunks": len(chunks),
            "num_legislations": len({c["serial"] for c in chunks}),
            "data_fingerprint": fingerprint,
            "built_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        }
        return cls(embeddings, chunks, meta)

    def save(self, index_dir: Optional[str] = None) -> None:
        index_dir = index_dir or INDEX_DIR
        os.makedirs(index_dir, exist_ok=True)
        np.save(os.path.join(index_dir, EMBEDDINGS_FILE), self.embeddings)
        with open(os.path.join(index_dir, CHUNKS_FILE), "w", encoding="utf-8") as f:
            json.dump(self.chunks, f, ensure_ascii=False, indent=1)
        with open(os.path.join(index_dir, META_FILE), "w", encoding="utf-8") as f:
            json.dump(self.meta, f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, index_dir: Optional[str] = None) -> "VectorStore":
        index_dir = index_dir or INDEX_DIR
        path = os.path.join(index_dir, EMBEDDINGS_FILE)
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"No vector index found in {index_dir}. Run: python build_index.py"
            )
        embeddings = np.load(path)
        with open(os.path.join(index_dir, CHUNKS_FILE), encoding="utf-8") as f:
            chunks = json.load(f)
        with open(os.path.join(index_dir, META_FILE), encoding="utf-8") as f:
            meta = json.load(f)
        return cls(embeddings, chunks, meta)

    def search(
        self,
        query_vector: np.ndarray,
        top_k: int = TOP_K,
        serial: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Return the top_k most similar chunks as dicts with a "score" field.
        Optionally restrict the search to one legislation (serial).
        """
        if len(self.chunks) == 0 or top_k <= 0:
            return []
        scores = self.embeddings @ query_vector.astype(np.float32)
        if serial:
            mask = np.array([c.get("serial") == serial for c in self.chunks])
            scores = np.where(mask, scores, -np.inf)
        order = np.argsort(-scores)[:top_k]
        return [
            {**self.chunks[i], "score": round(float(scores[i]), 4)}
            for i in order
            if np.isfinite(scores[i])
        ]
