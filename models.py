"""
Data models for legislation records.
"""

from dataclasses import dataclass
from typing import Optional, Dict, Any

# Fixed CSV column order (every row has every column, empty when missing)
CSV_FIELDS = [
    "serial",
    "title",
    "legislation_number",
    "legislation_type",
    "status",
    "issue_date",
    "publication_date",
    "update_date",
    "description",
    "source_url",
    "content",
    "classification",
    "decree",
    "valid_from_date",
]


@dataclass
class Legislation:
    """
    Represents a single legislation record extracted from the MOJ portal.
    Matches the required schema and only contains authentic fields.
    """

    serial: Optional[str] = None
    title: Optional[str] = None
    legislation_number: Optional[str] = None
    legislation_type: Optional[str] = None
    status: Optional[str] = None
    issue_date: Optional[str] = None
    publication_date: Optional[str] = None
    update_date: Optional[str] = None
    description: Optional[str] = None
    source_url: Optional[str] = None
    content: Optional[str] = None

    # Additional metadata fields actually present in official portal response
    classification: Optional[str] = None
    decree: Optional[str] = None
    valid_from_date: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dict, preserving non-empty fields in the specified order."""
        base_keys = [
            "serial",
            "title",
            "legislation_number",
            "legislation_type",
            "status",
            "issue_date",
            "publication_date",
            "update_date",
            "description",
            "source_url",
            "content",
        ]
        res = {}
        for k in base_keys:
            val = getattr(self, k)
            if val is not None:
                res[k] = val
        # Add optional metadata if present
        if self.classification:
            res["classification"] = self.classification
        if self.decree:
            res["decree"] = self.decree
        if self.valid_from_date:
            res["valid_from_date"] = self.valid_from_date
        return res

    def to_csv_dict(self) -> Dict[str, str]:
        """Convert to flat dict with empty strings for missing fields (for CSV output)."""
        return {k: ("" if getattr(self, k) is None else str(getattr(self, k))) for k in CSV_FIELDS}

    @property
    def unique_key(self) -> str:
        """Unique identifier for deduplication (the portal serial)."""
        return self.serial or ""
