from verify_integrity import load_sqlite, validate_records

URL = "https://laws.moj.gov.sa/ar/legislation/"


def good(serial="A", **kw):
    rec = {"serial": serial, "title": "نظام الإفلاس", "content": "المادة الأولى " * 10,
           "source_url": URL + serial, "issue_date": "1439-05-28"}
    rec.update(kw)
    return rec


def test_valid_records_pass():
    assert validate_records([good("A"), good("B")]) == {"errors": [], "warnings": []}


def test_detects_duplicates_empty_fields_and_bad_urls():
    report = validate_records([
        good("A"), good("A"),
        good("B", title=""),
        good("C", content=None),
        good("D", source_url=None),
        good("E", source_url="https://example.com/E"),
        good("", source_url=URL),
    ])
    text = "\n".join(report["errors"])
    assert "Duplicate serial A" in text
    assert "empty title" in text
    assert "empty content" in text
    assert "missing source_url" in text
    assert "malformed source_url https://example.com/E" in text
    assert "missing serial" in text


def test_detects_corrupted_arabic():
    mojibake = "نظام الإفلاس".encode("utf-8").decode("latin-1")
    report = validate_records([good("A", title=mojibake), good("B", content="نص � " * 10)])
    assert sum("corrupted Arabic" in e for e in report["errors"]) == 2


def test_missing_optional_fields_are_not_errors():
    report = validate_records([{"serial": "A", "title": "نظام", "content": "نص " * 30, "source_url": URL + "A"}])
    assert report["errors"] == []


def test_sample_database_reports_only_the_empty_record(db_path):
    report = validate_records(load_sqlite(db_path))
    assert report["errors"] == ["#3 لائحة بلا محتوى: empty content"]
