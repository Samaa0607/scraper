"""
Preprocessing and chunking of legislation text for RAG.

The scraper stores content as article-structured text:

    ## الفصل الأول - أحكام عامة          <- chapter heading
    ### المادة الأولى                    <- article heading
    نص المادة ...

Chunking strategy (simple, structure-aware):
1. Split the document on headings, keeping the current chapter as context.
2. Each article is one chunk if it fits in CHUNK_MAX_CHARS.
3. Long articles are split on paragraph (line) boundaries.
4. Very short consecutive articles in the same chapter are merged
   (up to CHUNK_MIN_CHARS) so chunks are not too tiny to be useful.

Every chunk keeps the legislation metadata plus its chapter/article heading.
"""

import re
from typing import Any, Dict, Iterable, List, Optional

from config import CHUNK_MAX_CHARS, CHUNK_MIN_CHARS

HEADING_RE = re.compile(r"^(#{1,6})\s+(.*)$")
# Invisible characters that only add noise (zero-width space, BOM, word joiner).
INVISIBLE_RE = re.compile(r"[​⁠﻿]")
SPACES_RE = re.compile(r"[ \t ]+")

METADATA_FIELDS = [
    "serial",
    "title",
    "legislation_type",
    "status",
    "issue_date",
    "publication_date",
    "source_url",
]


def clean_text(text: Optional[str]) -> str:
    """
    Normalize whitespace without altering legal wording.
    Arabic letters, diacritics and Arabic-Indic digits are preserved as-is.
    """
    if not text:
        return ""
    text = INVISIBLE_RE.sub("", text).replace("\r", "")
    lines = [SPACES_RE.sub(" ", line).strip() for line in text.split("\n")]
    # Collapse runs of blank lines to a single blank line
    cleaned = re.sub(r"\n{3,}", "\n\n", "\n".join(lines))
    return cleaned.strip()


def _split_sections(content: str) -> List[Dict[str, str]]:
    """Split content into sections: [{"chapter", "heading", "body"}]."""
    sections: List[Dict[str, str]] = []
    chapter = ""
    current: Optional[Dict[str, Any]] = None

    for line in content.split("\n"):
        match = HEADING_RE.match(line)
        if match:
            level, heading = len(match.group(1)), match.group(2).strip()
            if current:
                sections.append(current)
            # Level-2 headings ("##") are chapters/parts; deeper ones are articles.
            if level <= 2:
                chapter = heading
            current = {"chapter": chapter, "heading": heading, "lines": []}
        else:
            if current is None:
                current = {"chapter": "", "heading": "", "lines": []}
            if line.strip():
                current["lines"].append(line.strip())

    if current:
        sections.append(current)

    result = []
    for s in sections:
        body = "\n".join(s["lines"]).strip()
        # A chapter heading with no text of its own is context only, not a chunk
        if not body:
            continue
        result.append({"chapter": s["chapter"], "heading": s["heading"], "body": body})
    return result


def _split_long_text(text: str, max_chars: int) -> List[str]:
    """Split text on line boundaries into pieces of at most ~max_chars."""
    pieces: List[str] = []
    buf = ""
    for line in text.split("\n"):
        # A single line longer than max_chars is split on sentence/word boundaries
        while len(line) > max_chars:
            cut = max(line.rfind(" ", 0, max_chars), line.rfind("،", 0, max_chars), line.rfind(".", 0, max_chars))
            cut = cut if cut > max_chars // 2 else max_chars
            head, line = line[:cut].strip(), line[cut:].strip()
            if buf:
                pieces.append(buf)
                buf = ""
            pieces.append(head)
        if buf and len(buf) + 1 + len(line) > max_chars:
            pieces.append(buf)
            buf = line
        else:
            buf = f"{buf}\n{line}" if buf else line
    if buf:
        pieces.append(buf)
    return pieces


def chunk_legislation(
    record: Dict[str, Any],
    max_chars: int = CHUNK_MAX_CHARS,
    min_chars: int = CHUNK_MIN_CHARS,
) -> List[Dict[str, Any]]:
    """
    Split one legislation record into chunks with metadata.
    Returns [] for records without content.
    """
    content = clean_text(record.get("content"))
    if not content:
        return []

    metadata = {f: record.get(f) for f in METADATA_FIELDS}
    serial = record.get("serial") or "unknown"

    # 1. Section-level pieces (article, or part of a long article)
    pieces: List[Dict[str, str]] = []
    for section in _split_sections(content):
        parts = _split_long_text(section["body"], max_chars)
        for i, part in enumerate(parts):
            heading = section["heading"]
            if len(parts) > 1:
                heading = f"{heading} ({i + 1}/{len(parts)})" if heading else f"({i + 1}/{len(parts)})"
            pieces.append({"chapter": section["chapter"], "heading": heading, "text": part})

    # 2. Merge very short consecutive pieces of the same chapter
    merged: List[Dict[str, str]] = []
    for piece in pieces:
        block = f"{piece['heading']}\n{piece['text']}" if piece["heading"] else piece["text"]
        prev = merged[-1] if merged else None
        if (
            prev is not None
            and prev["chapter"] == piece["chapter"]
            and len(prev["body"]) < min_chars
            and len(prev["body"]) + len(block) + 2 <= max_chars
        ):
            prev["body"] += "\n\n" + block
            prev["headings"].append(piece["heading"])
        else:
            merged.append({"chapter": piece["chapter"], "headings": [piece["heading"]], "body": block})

    # 3. Build chunks with metadata. The title (and chapter) is prepended to the
    #    embedded text so each chunk is understandable on its own.
    chunks = []
    for i, m in enumerate(merged):
        headings = [h for h in m["headings"] if h]
        heading = headings[0] if len(headings) <= 1 else f"{headings[0]} … {headings[-1]}"
        context_lines = [metadata["title"] or ""]
        if m["chapter"] and m["chapter"] not in m["body"][: len(m["chapter"]) + 5]:
            context_lines.append(m["chapter"])
        chunks.append({
            "chunk_id": f"{serial}-{i:04d}",
            "chunk_index": i,
            "chapter": m["chapter"] or None,
            "heading": heading or None,
            "text": m["body"],
            "embed_text": "\n".join([l for l in context_lines if l] + [m["body"]]),
            **metadata,
        })
    return chunks


def chunk_records(records: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Chunk many legislation records, skipping those with empty content."""
    chunks: List[Dict[str, Any]] = []
    for record in records:
        chunks.extend(chunk_legislation(record))
    return chunks


def load_records_from_db(db_path: Optional[str] = None) -> List[Dict[str, Any]]:
    """Read all legislation records from SQLite as dicts."""
    from database import get_all_legislations
    return [dict(row) for row in get_all_legislations(db_path)]
