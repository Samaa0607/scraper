"""
Data-quality validation for scraped legislation (offline, no network).

Checks the SQLite database (and the JSON/CSV exports of the last run) for:
- duplicate serials
- empty titles
- empty content
- missing / malformed source URLs
- malformed records (missing serial, bad date format)
- corrupted Arabic text (mojibake, replacement characters, no Arabic letters)

Optional fields (status, dates, description, ...) are NOT required; a missing
optional field is never reported as an error.

Usage:
    python verify_integrity.py
"""

import csv
import json
import os
import re
import sys
from collections import Counter
from typing import Any, Dict, List

from config import CSV_OUTPUT_FILE, DATABASE_PATH, JSON_OUTPUT_FILE, LEGISLATION_WEB_URL

ARABIC_LETTER_RE = re.compile(r"[؀-ۿ]")
# Typical mojibake when UTF-8 Arabic is decoded as Latin-1/CP1252 (e.g. "Ø§Ù„"),
# plus the Unicode replacement character.
MOJIBAKE_RE = re.compile(r"[ØÙÚÛ][\u0080-ÿ]|�")
DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")

MIN_CONTENT_CHARS = 50


def is_corrupted_arabic(text: str) -> bool:
    """True if the text shows signs of broken encoding."""
    return bool(text) and bool(MOJIBAKE_RE.search(text))


def validate_records(records: List[Dict[str, Any]]) -> Dict[str, List[str]]:
    """
    Validate a list of legislation dicts.
    Returns {"errors": [...], "warnings": [...]}. Errors are real data problems;
    warnings are worth a look but do not make the data unusable.
    """
    errors: List[str] = []
    warnings: List[str] = []

    serial_counts = Counter((r.get("serial") or "").strip() for r in records)
    for serial, count in serial_counts.items():
        if serial and count > 1:
            errors.append(f"Duplicate serial {serial} ({count} records)")

    for i, r in enumerate(records, 1):
        serial = (r.get("serial") or "").strip()
        title = (r.get("title") or "").strip()
        content = (r.get("content") or "").strip()
        url = (r.get("source_url") or "").strip()
        label = f"#{i} {title[:40] or serial or '?'}"

        if not serial:
            errors.append(f"{label}: missing serial")
        if not title:
            errors.append(f"{label}: empty title")
        elif not ARABIC_LETTER_RE.search(title):
            warnings.append(f"{label}: title contains no Arabic letters")

        if not content:
            errors.append(f"{label}: empty content")
        elif len(content) < MIN_CONTENT_CHARS:
            warnings.append(f"{label}: very short content ({len(content)} chars)")

        if not url:
            errors.append(f"{label}: missing source_url")
        elif not url.startswith(f"{LEGISLATION_WEB_URL}/") or (serial and not url.endswith(serial)):
            errors.append(f"{label}: malformed source_url {url}")

        for field in ("title", "description", "content"):
            if is_corrupted_arabic(r.get(field) or ""):
                errors.append(f"{label}: corrupted Arabic text in '{field}'")

        for field in ("issue_date", "publication_date"):
            value = r.get(field)
            if value and not DATE_RE.match(str(value)):
                warnings.append(f"{label}: unexpected {field} format {value!r}")

    return {"errors": errors, "warnings": warnings}


def load_sqlite(path: str = DATABASE_PATH) -> List[Dict[str, Any]]:
    from database import get_all_legislations
    return [dict(row) for row in get_all_legislations(path)]


def load_json(path: str = JSON_OUTPUT_FILE) -> List[Dict[str, Any]]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def load_csv(path: str = CSV_OUTPUT_FILE) -> List[Dict[str, Any]]:
    with open(path, encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def main() -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

    sources = [
        ("SQLite", DATABASE_PATH, load_sqlite),
        ("JSON (last run)", JSON_OUTPUT_FILE, load_json),
        ("CSV (last run)", CSV_OUTPUT_FILE, load_csv),
    ]

    print("=" * 70)
    print("  DATA-QUALITY VALIDATION")
    print("=" * 70)

    total_errors = 0
    for name, path, loader in sources:
        if not os.path.exists(path):
            print(f"\n[{name}] not found: {path}")
            if name == "SQLite":
                total_errors += 1
            continue

        records = loader(path)
        report = validate_records(records)
        with_content = sum(1 for r in records if (r.get("content") or "").strip())
        print(f"\n[{name}] {path}")
        print(f"  records: {len(records)} | with content: {with_content} | "
              f"errors: {len(report['errors'])} | warnings: {len(report['warnings'])}")
        for e in report["errors"]:
            print(f"  ERROR   {e}")
        for w in report["warnings"]:
            print(f"  WARNING {w}")
        total_errors += len(report["errors"])

    print("\n" + "=" * 70)
    print(f"  RESULT: {'PASS' if total_errors == 0 else 'FAIL'} ({total_errors} errors)")
    print("=" * 70)
    return 0 if total_errors == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
