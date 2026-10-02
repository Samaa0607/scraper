"""
Build (or rebuild) the RAG vector index from the SQLite database.

    SQLite -> clean + chunk -> embed -> data/index/

Usage:
    python build_index.py            # rebuild only if the data changed
    python build_index.py --force    # always rebuild
"""

import argparse
import logging
import sys

from config import INDEX_DIR, LOG_FORMAT
from rag.chunking import chunk_records, load_records_from_db
from rag.vector_store import Embedder, VectorStore, data_fingerprint

logger = logging.getLogger("build_index")


def build_index(force: bool = False, index_dir: str = None, db_path=None, embedder=None) -> VectorStore:
    index_dir = index_dir or INDEX_DIR
    records = load_records_from_db(db_path)
    with_content = [r for r in records if (r.get("content") or "").strip()]
    skipped = len(records) - len(with_content)
    fingerprint = data_fingerprint(with_content)

    if not force:
        try:
            existing = VectorStore.load(index_dir)
            if existing.meta.get("data_fingerprint") == fingerprint:
                logger.info(f"Index is up to date ({len(existing)} chunks). Use --force to rebuild.")
                return existing
            logger.info("Source data changed since last build; rebuilding index.")
        except FileNotFoundError:
            logger.info("No existing index; building a new one.")

    logger.info(f"Legislations: {len(records)} total, {len(with_content)} with content, {skipped} skipped (empty content)")
    chunks = chunk_records(with_content)
    logger.info(f"Chunks: {len(chunks)}")

    store = VectorStore.build(chunks, embedder or Embedder(), fingerprint)
    store.save(index_dir)
    logger.info(f"Saved index to {index_dir} ({store.meta['num_chunks']} chunks, model={store.meta['embedding_model']})")
    return store


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    logging.basicConfig(level=logging.INFO, format=LOG_FORMAT, stream=sys.stdout)

    parser = argparse.ArgumentParser(description="Build the RAG vector index from SQLite")
    parser.add_argument("--force", action="store_true", help="Rebuild even if the data is unchanged")
    args = parser.parse_args()
    build_index(force=args.force)


if __name__ == "__main__":
    main()
