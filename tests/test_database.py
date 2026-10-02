import database
from tests.conftest import make_legislation


def test_arabic_round_trip_and_count(db_path):
    rows = database.get_all_legislations(db_path)
    assert len(rows) == 3
    assert rows[0]["title"] == "نظام الإفلاس"
    assert "التسوية الوقائية" in rows[0]["content"]


def test_reinsert_does_not_duplicate_and_keeps_id(db_path, sample_legislations):
    before = {r["serial"]: r["id"] for r in database.get_all_legislations(db_path)}
    database.insert_legislations(sample_legislations, db_path)
    rows = database.get_all_legislations(db_path)
    assert len(rows) == 3
    assert {r["serial"]: r["id"] for r in rows} == before


def test_empty_content_does_not_overwrite_existing_content(db_path):
    database.insert_legislation(make_legislation("SER-BANK", "نظام الإفلاس", None), db_path)
    row = database.get_legislation_by_serial("SER-BANK", db_path)
    assert row["content"] and "المادة الأولى" in row["content"]


def test_new_content_fills_previously_empty_record(db_path):
    database.insert_legislation(make_legislation("SER-EMPTY", "لائحة بلا محتوى", "### المادة الأولى\nنص"), db_path)
    assert database.get_legislation_by_serial("SER-EMPTY", db_path)["content"] == "### المادة الأولى\nنص"


def test_malformed_records_are_skipped(db_path):
    written = database.insert_legislations(
        [make_legislation(None, "بدون رقم", "x"), make_legislation("S2", None, "x")], db_path
    )
    assert written == 0
    assert database.get_legislation_count(db_path) == 3


def test_search_legislations_ranks_title_matches_first(db_path):
    rows = database.search_legislations("الإفلاس", db_path=db_path)
    assert rows[0]["serial"] == "SER-BANK"
    assert database.search_legislations("غير موجود إطلاقا", db_path=db_path) == []
