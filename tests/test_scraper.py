"""Offline scraper tests: the browser page is replaced by a fake that serves canned API responses."""

import asyncio
import csv
import json
import sqlite3

import pytest

import scraper as scraper_module
from scraper import MOJScraper


def _item(serial, name):
    return {"serial": serial, "statuteName": name, "legalType": "نظام"}


def _detail(serial, name):
    return {"serial": serial, "name": name, "statuteStructure": [
        {"sequence": "المادة الأولى", "text": f"<p>نص {name}</p>", "items": []}]}


class FakePage:
    """Mimics page.evaluate(FETCH_JSON_JS, {...}) using a dict of responses."""

    def __init__(self, search_pages, details, fail_detail_times=None):
        self.search_pages = search_pages          # {(by_title, page): model}
        self.details = details                    # {serial: model}
        self.fail_detail_times = dict(fail_detail_times or {})
        self.calls = []

    async def evaluate(self, script, arg=None):
        if arg is None:
            return None
        url, body = arg["url"], arg["body"]
        self.calls.append(url)
        if body is not None:
            key = (bool(body["statuteName"]), body["pageNumber"])
            return {"success": True, "model": self.search_pages.get(key, {"collection": [], "totalPages": 1})}
        serial = url.split("Serial=")[1].split("&")[0]
        if self.fail_detail_times.get(serial, 0) > 0:
            self.fail_detail_times[serial] -= 1
            raise RuntimeError("TypeError: Failed to fetch")
        if serial not in self.details:
            return {"success": False, "message": "not found", "model": None}
        return {"success": True, "model": self.details[serial]}


@pytest.fixture(autouse=True)
def no_retry_wait(monkeypatch):
    monkeypatch.setattr(scraper_module, "RETRY_DELAYS_SECONDS", [0, 0, 0])


def make_scraper(page):
    s = MOJScraper(delay=0)
    s._page = page
    return s


def test_search_dedupes_retries_and_counts_empty_content():
    page = FakePage(
        search_pages={
            (True, 1): {"collection": [_item("A", "نظام الإفلاس"), _item("B", "نظام العمل")], "totalCount": 2, "totalPages": 1},
            (False, 1): {"collection": [_item("A", "نظام الإفلاس"), {"statuteName": "بدون رقم"}], "totalCount": 3, "totalPages": 2},
            (False, 2): {"collection": [_item("C", "نظام التنفيذ")], "totalCount": 3, "totalPages": 2},
        },
        details={"A": _detail("A", "نظام الإفلاس"), "B": _detail("B", "نظام العمل")},  # C has no detail
        fail_detail_times={"A": 2},  # two transient failures, third attempt succeeds
    )
    s = make_scraper(page)
    results = asyncio.run(s.search("الإفلاس"))

    assert [r.serial for r in results] == ["A", "B", "C"]
    assert results[0].content and "نص نظام الإفلاس" in results[0].content  # recovered by retry
    assert results[2].content is None
    assert s.stats["duplicates"] == 1
    assert s.stats["malformed_items"] == 1
    assert s.stats["detail_failures"] == 1
    assert s.stats["empty_content"] == 1
    assert s.stats["pages_processed"] == 3
    assert s.stats["pages_failed"] == 0


def test_search_respects_max_results():
    page = FakePage(
        search_pages={(True, 1): {"collection": [_item(str(i), f"نظام {i}") for i in range(5)], "totalPages": 1}},
        details={str(i): _detail(str(i), f"نظام {i}") for i in range(5)},
    )
    results = asyncio.run(make_scraper(page).search("نظام", max_results=2))
    assert len(results) == 2


def test_failed_first_page_aborts_pass_quickly():
    class DownPage(FakePage):
        async def evaluate(self, script, arg=None):
            raise RuntimeError("TypeError: Failed to fetch")

    s = make_scraper(DownPage({}, {}))
    assert asyncio.run(s.search("الإفلاس")) == []
    assert s.stats["pages_failed"] == 2  # one per pass, no 50-page loop


def test_save_results_writes_json_csv_sqlite_with_arabic(tmp_path, sample_legislations):
    s = MOJScraper(delay=0)
    paths = {k: str(tmp_path / f"out.{k}") for k in ("json", "csv", "db")}
    s.save_results(sample_legislations, json_path=paths["json"], csv_path=paths["csv"], db_path=paths["db"])

    data = json.load(open(paths["json"], encoding="utf-8"))
    assert data[0]["title"] == "نظام الإفلاس"
    assert "نظام الإفلاس" in open(paths["json"], encoding="utf-8").read()  # not \u-escaped

    rows = list(csv.DictReader(open(paths["csv"], encoding="utf-8-sig")))
    assert len(rows) == 3 and rows[1]["title"] == "نظام العمل"
    assert rows[2]["content"] == ""

    con = sqlite3.connect(paths["db"])
    assert con.execute("SELECT COUNT(*) FROM legislations").fetchone()[0] == 3
    assert con.execute("SELECT title FROM legislations WHERE serial='SER-BANK'").fetchone()[0] == "نظام الإفلاس"
