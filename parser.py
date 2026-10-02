"""
Parser and data extraction helpers for MOJ legislation records.
Cleans raw text, extracts legislation numbers, formats dates,
and recursively formats the full legal articles and chapters.
"""

import re
import logging
from typing import Optional, List, Dict, Any
from bs4 import BeautifulSoup

from models import Legislation
from config import LEGISLATION_WEB_URL

logger = logging.getLogger(__name__)


def _text(value: Any) -> str:
    """Return a stripped string for any API value (None/numbers safe)."""
    if value is None:
        return ""
    return str(value).strip()


def clean_html_text(raw_html: Optional[str]) -> str:
    """
    Safely clean HTML markup from legislation text while preserving paragraph breaks.
    Normalizes whitespace and non-breaking spaces without altering legal words.
    """
    if not raw_html:
        return ""

    soup = BeautifulSoup(raw_html, "html.parser")
    # Insert newlines before block elements to maintain readable formatting
    for tag in soup.find_all(["p", "br", "div", "li", "tr"]):
        tag.insert_before("\n")

    text = soup.get_text()
    # Normalize non-breaking spaces and carriage returns
    text = text.replace("\xa0", " ").replace("\r", "")
    lines = [line.strip() for line in text.split("\n")]
    cleaned_lines = [line for line in lines if line]
    return "\n".join(cleaned_lines)


def format_date(date_str: Optional[str]) -> Optional[str]:
    """
    Format date string (e.g., '1440-09-21T00:00:00' -> '1440-09-21').
    Returns None if date is empty or invalid.
    """
    date_str = _text(date_str)
    if not date_str:
        return None
    match = re.match(r"^(\d{4}[-/]\d{1,2}[-/]\d{1,2})", date_str)
    if match:
        return match.group(1).replace("/", "-")
    return date_str


def extract_legislation_number(
    title: Optional[str],
    decree: Optional[str],
    summary: Optional[str]
) -> Optional[str]:
    """
    Extract legislation number (e.g., 'م/50' or '512' or 'م/164')
    from decree, title, or summary text.
    """
    search_texts = [decree or "", title or "", summary or ""]

    patterns = [
        r"(?:مرسوم\s*ملكي\s*رقم|أمر\s*ملكي\s*رقم|رقم\s*)\s*([مM]/\s*\d+)",
        r"(?:قرار\s*وزير\s*العدل\s*رقم|قرار\s*مجلس\s*الوزراء\s*رقم|قرار\s*رقم)\s*\(?([0-9]+)\)?",
        r"\b([مM]/\s*\d+)\b",
    ]

    for text in search_texts:
        if not text:
            continue
        for pattern in patterns:
            match = re.search(pattern, text)
            if match:
                return match.group(1).strip()

    return None


def _extract_structure_node(node: Dict[str, Any], depth: int = 1) -> List[str]:
    """
    Recursively extract heading and body text from a statuteStructure node and its children.
    Handles chapters (الباب / الفصل), articles (المادة), and nested sub-items.
    """
    blocks = []
    seq = _text(node.get("sequence"))
    name = _text(node.get("name"))
    decree = _text(node.get("decree"))
    raw_text = _text(node.get("text"))
    clean_text = clean_html_text(raw_text)

    # Build heading
    heading_parts = []
    if seq:
        heading_parts.append(seq)
    if name and name != seq:
        heading_parts.append(name)
    if decree:
        heading_parts.append(f"({decree})")
    heading = " - ".join(heading_parts)

    prefix = "#" * min(depth + 1, 5)

    if heading and clean_text:
        blocks.append(f"{prefix} {heading}\n{clean_text}")
    elif heading:
        blocks.append(f"{prefix} {heading}")
    elif clean_text:
        blocks.append(clean_text)

    # Recursively traverse nested child items (e.g. articles inside a chapter)
    child_items = node.get("items") or []
    for child in child_items:
        if not isinstance(child, dict):
            continue
        blocks.extend(_extract_structure_node(child, depth + 1))

    return blocks


def build_full_content(statute_structure: Optional[List[Dict[str, Any]]]) -> Optional[str]:
    """
    Reconstruct the full legislation text by traversing the entire article tree.
    Extracts all chapters, sections, and individual articles verbatim.
    """
    if not statute_structure:
        return None

    all_blocks = []
    for top_node in statute_structure:
        if not isinstance(top_node, dict):
            continue
        all_blocks.extend(_extract_structure_node(top_node, depth=1))

    if not all_blocks:
        return None

    return "\n\n".join(all_blocks)


def parse_legislation_data(
    search_item: Dict[str, Any],
    detail_model: Optional[Dict[str, Any]] = None
) -> Legislation:
    """
    Combine search result metadata and full detail model into a unified Legislation object.
    Only extracts authentic fields from the official MOJ portal response.
    """
    model = detail_model or {}

    title = _text(model.get("name") or search_item.get("statuteName"))

    serial = _text(model.get("serial") or search_item.get("serial"))
    source_url = f"{LEGISLATION_WEB_URL}/{serial}" if serial else None

    leg_type = _text(model.get("legalType") or search_item.get("legalType")) or None

    status = _text(model.get("legalStatueName") or search_item.get("legalStatueName")) or None

    issue_date = format_date(
        model.get("issuanceDate")
        or search_item.get("issueDate")
        or search_item.get("issuanceDate")
    )
    publication_date = format_date(
        model.get("publishDate")
        or search_item.get("publishDate")
    )
    update_date = format_date(
        model.get("validFromDate")
        or search_item.get("validFromDate")
    )

    description = (
        model.get("summary")
        or search_item.get("summary")
        or None
    )
    if description:
        description = clean_html_text(_text(description)) or None

    decree = None
    structure = model.get("statuteStructure") or []
    if structure and isinstance(structure[0], dict):
        # Check first top node or its first child for decree
        first_decree = _text(structure[0].get("decree"))
        first_items = structure[0].get("items") or []
        if not first_decree and first_items and isinstance(first_items[0], dict):
            first_decree = _text(first_items[0].get("decree"))
        decree = first_decree or None

    leg_number = extract_legislation_number(title, decree, description)

    # Reconstruct full legal content from all articles
    content = build_full_content(structure)

    classification = _text(model.get("classificationName")) or None

    return Legislation(
        serial=serial or None,
        title=title or None,
        legislation_number=leg_number,
        legislation_type=leg_type,
        status=status,
        issue_date=issue_date,
        publication_date=publication_date,
        update_date=update_date,
        description=description,
        source_url=source_url,
        content=content,
        classification=classification,
        decree=decree,
        valid_from_date=update_date,
    )
