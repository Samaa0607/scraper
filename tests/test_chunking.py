import re

from rag.chunking import chunk_legislation, chunk_records, clean_text
from tests.conftest import BANKRUPTCY_CONTENT


def record(content, serial="S1"):
    return {"serial": serial, "title": "نظام الإفلاس", "legislation_type": "نظام", "status": "ساري",
            "issue_date": "1439-05-28", "publication_date": "1439-06-14",
            "source_url": f"https://laws.moj.gov.sa/ar/legislation/{serial}", "content": content}


def test_clean_text_normalizes_whitespace_and_keeps_arabic():
    raw = "المادة​  الأولى\t نص\r\n\n\n\nسطر   ثانٍ ١٢٣"
    assert clean_text(raw) == "المادة الأولى نص\n\nسطر ثانٍ ١٢٣"


def test_empty_content_yields_no_chunks():
    assert chunk_legislation(record(None)) == []
    assert chunk_legislation(record("   ")) == []
    assert chunk_records([record(None), record(BANKRUPTCY_CONTENT, "S2")])[0]["serial"] == "S2"


def test_chunks_keep_metadata_and_article_headings():
    chunks = chunk_legislation(record(BANKRUPTCY_CONTENT), min_chars=0)
    assert len(chunks) == 3
    first = chunks[0]
    for field in ("serial", "title", "legislation_type", "status", "issue_date", "publication_date", "source_url"):
        assert first[field], field
    assert first["chunk_id"] == "S1-0000"
    assert first["chapter"] == "الفصل الأول - أحكام عامة"
    assert first["heading"] == "المادة الأولى"
    assert first["text"].startswith("المادة الأولى")
    assert chunks[2]["chapter"] == "الفصل الثاني - التصفية"
    # title + chapter are added to the embedded text for context
    assert "نظام الإفلاس" in first["embed_text"] and "أحكام عامة" in first["embed_text"]


def test_short_articles_are_merged_within_a_chapter_only():
    chunks = chunk_legislation(record(BANKRUPTCY_CONTENT), min_chars=1000)
    assert len(chunks) == 2  # articles 1+2 merged; article 3 is in another chapter
    assert "المادة الثانية" in chunks[0]["text"]


def test_long_article_is_split_without_losing_text():
    long_body = "\n".join(f"فقرة رقم {i} " + "نص نظامي طويل " * 20 for i in range(30))
    content = f"### المادة الأولى\n{long_body}"
    chunks = chunk_legislation(record(content), max_chars=800)
    assert len(chunks) > 1
    assert all(len(c["text"]) <= 800 + 40 for c in chunks)
    assert chunks[0]["heading"].startswith("المادة الأولى (1/")
    original = set(re.findall(r"\w+", long_body))
    combined = set(re.findall(r"\w+", " ".join(c["text"] for c in chunks)))
    assert original <= combined
