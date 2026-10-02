"""
SQLite storage for legislation records.
"""

import logging
import os
import sqlite3
from typing import List, Optional

from config import DATABASE_PATH

logger = logging.getLogger(__name__)


# Upsert keyed on serial. Unlike INSERT OR REPLACE, this keeps the row id stable
# and never overwrites previously stored content with an empty value (e.g. when a
# later run fails to fetch the detail API for a legislation).
UPSERT_SQL = """
    INSERT INTO legislations (
        serial, title, legislation_type, status, issue_date,
        publication_date, description, source_url, content
    )
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    ON CONFLICT(serial) DO UPDATE SET
        title            = excluded.title,
        legislation_type = COALESCE(excluded.legislation_type, legislations.legislation_type),
        status           = COALESCE(excluded.status, legislations.status),
        issue_date       = COALESCE(excluded.issue_date, legislations.issue_date),
        publication_date = COALESCE(excluded.publication_date, legislations.publication_date),
        description      = COALESCE(NULLIF(excluded.description, ''), legislations.description),
        source_url       = COALESCE(excluded.source_url, legislations.source_url),
        content          = COALESCE(NULLIF(excluded.content, ''), legislations.content)
"""


def get_connection(db_path: Optional[str] = None):
    """Create and return a SQLite database connection."""
    path = db_path or DATABASE_PATH
    os.makedirs(os.path.dirname(path), exist_ok=True)

    connection = sqlite3.connect(path)
    connection.row_factory = sqlite3.Row

    return connection


def create_tables(db_path: Optional[str] = None):
    """Create the legislations table if it does not exist."""

    connection = get_connection(db_path)
    try:
        connection.execute("""
            CREATE TABLE IF NOT EXISTS legislations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                serial TEXT UNIQUE,
                title TEXT NOT NULL,
                legislation_type TEXT,
                status TEXT,
                issue_date TEXT,
                publication_date TEXT,
                description TEXT,
                source_url TEXT,
                content TEXT
            )
        """)
        connection.commit()
    finally:
        connection.close()


def insert_legislation(legislation, db_path: Optional[str] = None) -> int:
    """Insert or update one legislation in the database."""
    return insert_legislations([legislation], db_path)


def insert_legislations(legislations: List, db_path: Optional[str] = None) -> int:
    """
    Insert or update multiple legislations in the database.
    Records without a serial or title are skipped (logged), since serial is the
    deduplication key and title is required by the schema.
    Returns the number of records written.
    """

    connection = get_connection(db_path)
    written = 0
    try:
        for legislation in legislations:
            data = legislation.to_dict()

            if not data.get("serial") or not data.get("title"):
                logger.warning(
                    f"Skipping DB insert for malformed record (serial={data.get('serial')!r}, "
                    f"title={data.get('title')!r})"
                )
                continue

            connection.execute(UPSERT_SQL, (
                data.get("serial"),
                data.get("title"),
                data.get("legislation_type"),
                data.get("status"),
                data.get("issue_date"),
                data.get("publication_date"),
                data.get("description"),
                data.get("source_url"),
                data.get("content"),
            ))
            written += 1

        connection.commit()
    finally:
        connection.close()

    return written


def get_all_legislations(db_path: Optional[str] = None):
    """Return all stored legislations."""

    connection = get_connection(db_path)
    try:
        return connection.execute("SELECT * FROM legislations ORDER BY id").fetchall()
    finally:
        connection.close()


def get_legislation_by_serial(serial: str, db_path: Optional[str] = None):
    """Return one legislation by serial, or None."""

    connection = get_connection(db_path)
    try:
        return connection.execute(
            "SELECT * FROM legislations WHERE serial = ?", (serial,)
        ).fetchone()
    finally:
        connection.close()


def search_legislations(query: str, limit: int = 10, db_path: Optional[str] = None):
    """
    Simple text search over title, description and content (SQL LIKE).
    Title matches are ranked first.
    """

    pattern = f"%{query.strip()}%"
    connection = get_connection(db_path)
    try:
        return connection.execute("""
            SELECT * FROM legislations
            WHERE title LIKE ? OR description LIKE ? OR content LIKE ?
            ORDER BY (title LIKE ?) DESC, id
            LIMIT ?
        """, (pattern, pattern, pattern, pattern, limit)).fetchall()
    finally:
        connection.close()


def get_legislation_count(db_path: Optional[str] = None):
    """Return number of legislations in the database."""

    connection = get_connection(db_path)
    try:
        return connection.execute("SELECT COUNT(*) FROM legislations").fetchone()[0]
    finally:
        connection.close()
