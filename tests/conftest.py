"""
Shared test fixtures. All tests run offline:
- a temporary SQLite database with a few Arabic legislation records
- a deterministic bag-of-words "embedder" instead of the real model
- the LLM is mocked where needed
"""

import hashlib
import os
import re
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import Legislation  # noqa: E402

BANKRUPTCY_CONTENT = """## الفصل الأول - أحكام عامة

### المادة الأولى
يقصد بالتسوية الوقائية إجراء يهدف إلى تيسير توصل المدين إلى اتفاق مع دائنيه على تسوية ديونه مع احتفاظ المدين بإدارة نشاطه.

### المادة الثانية
يجوز للمدين أن يتقدم إلى المحكمة بطلب افتتاح إجراء التسوية الوقائية إذا كان يتوقع أن يعاني من اضطراب أوضاعه المالية.

## الفصل الثاني - التصفية

### المادة الثالثة
تهدف التصفية إلى بيع أصول المدين المفلس وتوزيع حصيلتها على الدائنين تحت إدارة أمين التصفية."""

LABOR_CONTENT = """## الباب الأول - التعريفات

### المادة الأولى
يستحق العامل إجازة سنوية لا تقل مدتها عن واحد وعشرين يومًا بأجر يدفع مقدمًا.

### المادة الثانية
لا يجوز لصاحب العمل تشغيل العامل أكثر من ثماني ساعات في اليوم الواحد."""


def make_legislation(serial, title, content, **kw):
    return Legislation(
        serial=serial,
        title=title,
        legislation_type=kw.get("legislation_type", "نظام"),
        status=kw.get("status", "ساري"),
        issue_date=kw.get("issue_date", "1439-05-28"),
        publication_date=kw.get("publication_date", "1439-06-14"),
        description=kw.get("description", "ملخص " + (title or "")),
        source_url=f"https://laws.moj.gov.sa/ar/legislation/{serial}",
        content=content,
    )


@pytest.fixture
def sample_legislations():
    return [
        make_legislation("SER-BANK", "نظام الإفلاس", BANKRUPTCY_CONTENT),
        make_legislation("SER-LABOR", "نظام العمل", LABOR_CONTENT),
        make_legislation("SER-EMPTY", "لائحة بلا محتوى", None),
    ]


@pytest.fixture
def db_path(tmp_path, sample_legislations, monkeypatch):
    """A temporary database populated with the sample records; also set as the default DB."""
    import database
    path = str(tmp_path / "test.db")
    monkeypatch.setattr(database, "DATABASE_PATH", path)
    database.create_tables(path)
    database.insert_legislations(sample_legislations, path)
    return path


class FakeEmbedder:
    """Deterministic hashed bag-of-words embeddings (Arabic-aware tokenization)."""

    model_name = "fake-bow"
    dim = 512

    def _vec(self, text):
        v = np.zeros(self.dim, dtype=np.float32)
        for tok in re.findall(r"\w+", text):
            # crude Arabic normalization: strip the definite article
            tok = re.sub(r"^(وال|بال|لل|ال)", "", tok)
            if len(tok) < 2:
                continue
            v[int(hashlib.md5(tok.encode()).hexdigest(), 16) % self.dim] += 1.0
        n = np.linalg.norm(v)
        return v / n if n else v

    def encode_passages(self, texts):
        return np.stack([self._vec(t) for t in texts]) if texts else np.zeros((0, self.dim), np.float32)

    def encode_query(self, text):
        return self._vec(text)


@pytest.fixture
def fake_embedder():
    return FakeEmbedder()


@pytest.fixture
def rag_pipeline(db_path, fake_embedder, monkeypatch):
    from rag import llm
    from rag.chunking import chunk_records, load_records_from_db
    from rag.rag import RAGPipeline
    from rag.vector_store import VectorStore

    monkeypatch.setattr(llm, "is_available", lambda: False)
    chunks = chunk_records(load_records_from_db(db_path))
    store = VectorStore.build(chunks, fake_embedder, "fp")
    return RAGPipeline(store=store, embedder=fake_embedder, min_score=0.15)
