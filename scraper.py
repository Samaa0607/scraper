"""
Web scraper for the Saudi Ministry of Justice Laws & Regulations portal.

Uses Playwright to maintain the legitimate browser session (with security cookies/tokens)
and queries the official public API endpoints used by the MOJ Nuxt portal:
- Search API: POST https://laws-gateway.moj.gov.sa/apis/legislations/v1/statute/section-search
- Detail API: GET  https://laws-gateway.moj.gov.sa/apis/legislations/v1/statute/get-Statute-gateway-Detail
"""

import asyncio
import csv
import json
import logging
import os
from typing import List, Optional, Set, Dict, Any

from playwright.async_api import async_playwright, Browser, Page, BrowserContext

from config import (
    PORTAL_URL,
    SEARCH_API_URL,
    DETAIL_API_URL,
    DEFAULT_PAGE_SIZE,
    DEFAULT_SORTING,
    DEFAULT_MAX_PAGES,
    REQUEST_DELAY_SECONDS,
    PAGE_LOAD_TIMEOUT_MS,
    MAX_RETRIES,
    RETRY_DELAYS_SECONDS,
    OUTPUT_DIR,
    JSON_OUTPUT_FILE,
    CSV_OUTPUT_FILE,
    DATABASE_PATH,
    HEADLESS,
    VIEWPORT_WIDTH,
    VIEWPORT_HEIGHT,
)
from models import Legislation, CSV_FIELDS
from parser import parse_legislation_data
from database import create_tables, insert_legislations

logger = logging.getLogger(__name__)


# Creates a pristine native fetch from a hidden iframe to bypass Dynatrace interference
INSTALL_NATIVE_FETCH_JS = """() => {
    if (!window._nativeFetch) {
        const iframe = document.createElement('iframe');
        iframe.style.display = 'none';
        document.body.appendChild(iframe);
        window._nativeFetch = iframe.contentWindow.fetch.bind(window);
    }
}"""

# Generic JSON request executed inside the browser context (session/WAF continuity).
# Arguments are passed as data (never interpolated into the script), so queries
# containing quotes or other special characters are safe.
FETCH_JSON_JS = """async ({url, method, body}) => {
    const options = {
        method: method,
        headers: {'Accept': 'application/json, text/plain, */*'}
    };
    // Only send Content-Type with a body: on GET it would trigger a CORS
    // preflight that the detail endpoint rejects ("Failed to fetch").
    if (body !== null) {
        options.headers['Content-Type'] = 'application/json';
        options.body = JSON.stringify(body);
    }
    let res;
    try {
        res = await (window._nativeFetch || window.fetch)(url, options);
    } catch (err) {
        if (window._nativeFetch && window._nativeFetch !== window.fetch) {
            res = await window.fetch(url, options);
        } else {
            throw err;
        }
    }
    if (!res.ok) {
        throw new Error('HTTP ' + res.status + ' ' + res.statusText);
    }
    return await res.json();
}"""


def _short_error(e: Exception) -> str:
    err_str = str(e)
    if "Failed to fetch" in err_str:
        return "Failed to fetch"
    return err_str.split("\n")[0]


class MOJScraper:
    """
    Scraper for the official Saudi Ministry of Justice Laws & Regulations portal.
    """

    def __init__(
        self,
        headless: bool = HEADLESS,
        max_pages: int = DEFAULT_MAX_PAGES,
        delay: float = REQUEST_DELAY_SECONDS,
    ):
        self.headless = headless
        self.max_pages = max_pages
        self.delay = delay
        self._playwright = None
        self._browser: Optional[Browser] = None
        self._context: Optional[BrowserContext] = None
        self._page: Optional[Page] = None
        self._seen_serials: Set[str] = set()

        # Scraping and pagination statistics
        self.stats = {
            "expected_records": 0,
            "pages_processed": 0,
            "pages_failed": 0,
            "records_received": 0,
            "unique_legislations": 0,
            "duplicates": 0,
            "malformed_items": 0,
            "detail_failures": 0,
            "parse_failures": 0,
            "empty_content": 0,
        }
        self.failed_pages: List[str] = []
        self.failed_details: List[str] = []
        self.empty_content_titles: List[str] = []
        self.output_locations: Dict[str, str] = {}

    async def __aenter__(self):
        """Initialize Playwright browser and establish session."""
        self._playwright = await async_playwright().start()
        self._browser = await self._playwright.chromium.launch(
            headless=self.headless,
            args=["--disable-blink-features=AutomationControlled"],
        )
        self._context = await self._browser.new_context(
            viewport={"width": VIEWPORT_WIDTH, "height": VIEWPORT_HEIGHT},
            locale="ar-SA",
            user_agent=(
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/124.0.0.0 Safari/537.36"
            ),
        )
        self._page = await self._context.new_page()
        self._page.set_default_timeout(PAGE_LOAD_TIMEOUT_MS)

        logger.info(f"Navigating to {PORTAL_URL} to initialize session...")
        try:
            await self._page.goto(PORTAL_URL, wait_until="domcontentloaded", timeout=45000)
            await asyncio.sleep(2.5)
        except Exception as e:
            logger.warning(f"Navigation warning (proceeding with session): {e}")

        # The portal may still redirect/re-render after DOMContentLoaded, which destroys
        # the JS execution context. Wait for the page to settle and retry if needed.
        try:
            for attempt in range(1, MAX_RETRIES + 1):
                try:
                    await self._page.wait_for_load_state("load", timeout=PAGE_LOAD_TIMEOUT_MS)
                    await self._page.evaluate(INSTALL_NATIVE_FETCH_JS)
                    break
                except Exception as e:
                    if attempt == MAX_RETRIES:
                        raise
                    logger.warning(f"Session setup attempt {attempt} failed ({_short_error(e)}), retrying...")
                    await asyncio.sleep(2)
        except Exception:
            await self.__aexit__(None, None, None)
            raise

        logger.info("Portal session initialized successfully.")
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Cleanup browser resources."""
        if self._context:
            await self._context.close()
        if self._browser:
            await self._browser.close()
        if self._playwright:
            await self._playwright.stop()
        logger.info("Browser session closed.")

    async def _rate_limit(self):
        """Respectful delay between requests."""
        if self.delay > 0:
            await asyncio.sleep(self.delay)

    async def _request_json(
        self,
        url: str,
        label: str,
        method: str = "GET",
        body: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Call a gateway API through the browser context and return its `model` payload.
        Retries temporary failures (network errors, HTTP errors, empty/unsuccessful
        responses) up to MAX_RETRIES times with progressive backoff.
        Raises the last error if all attempts fail.
        """
        last_exception: Optional[Exception] = None

        for attempt in range(1, MAX_RETRIES + 1):
            try:
                data = await self._page.evaluate(
                    FETCH_JSON_JS, {"url": url, "method": method, "body": body}
                )
                if not isinstance(data, dict):
                    raise ValueError("Empty or non-JSON API response")
                if data.get("success") is False:
                    raise ValueError(f"API reported failure: {data.get('message') or data.get('errors')}")
                model = data.get("model")
                if not isinstance(model, dict):
                    raise ValueError("API response has no 'model' object")
                return model
            except Exception as e:
                last_exception = e
                logger.warning(f"{label} failed (attempt {attempt}/{MAX_RETRIES}): {_short_error(e)}")
                if attempt < MAX_RETRIES:
                    delay = RETRY_DELAYS_SECONDS[min(attempt - 1, len(RETRY_DELAYS_SECONDS) - 1)]
                    logger.info(f"Retrying {label} after {delay} seconds...")
                    # Re-ensure iframe for pristine native fetch if the page was reset
                    try:
                        await self._page.evaluate(INSTALL_NATIVE_FETCH_JS)
                    except Exception:
                        pass
                    await asyncio.sleep(delay)

        raise last_exception

    async def _fetch_search_page(
        self,
        query: str,
        page_number: int,
        search_by_title: bool = False,
        pass_name: str = "search",
    ) -> Dict[str, Any]:
        """
        Fetch one page of search results.
        search_by_title: if True, sets statuteName; if False, sets keyword.
        """
        body = {
            "pageNumber": page_number,
            "pageSize": DEFAULT_PAGE_SIZE,
            "keyword": "" if search_by_title else query,
            "detailsKeyword": "",
            "LegalStatue": None,
            "classificationId": None,
            "sortingBy": DEFAULT_SORTING,
            "statuteIssueDateFrom": None,
            "statuteIssueDateTo": None,
            "statuteName": query if search_by_title else "",
            "statutePublishDateFrom": None,
            "statutePublishDateTo": None,
            "statuteType": None,
            "isSearch": True,
            "identityNumber": "",
        }
        return await self._request_json(
            SEARCH_API_URL, f"{pass_name} page {page_number}", method="POST", body=body
        )

    async def _fetch_statute_detail(self, serial: str) -> Optional[Dict[str, Any]]:
        """
        Fetch full details and legislation articles via the portal detail API.
        Returns None (and logs the error) if all retries fail.
        """
        detail_url = f"{DETAIL_API_URL}?Serial={serial}&identityNumber="
        try:
            return await self._request_json(detail_url, f"detail {serial}")
        except Exception as e:
            logger.error(f"[ERROR] Failed to fetch detail for serial {serial} after {MAX_RETRIES} attempts: {_short_error(e)}")
            return None

    async def search(self, query: str, max_results: Optional[int] = None) -> List[Legislation]:
        """
        Search legislation records matching query across all pages up to max_results.
        Performs a title-focused pass first to capture exact legislations, followed
        by a broader keyword pass if additional results are needed.
        """
        results: List[Legislation] = []

        logger.info(f"Starting legislation search for query: '{query}'")
        logger.debug(f"max_results={max_results}, max_pages={self.max_pages}")

        # Two search passes:
        # Pass 1: Title search (exact legislations named with the keyword)
        # Pass 2: Full-text / section search (legislations containing the keyword)
        passes = [
            ("title search", True),
            ("keyword search", False),
        ]

        for pass_name, search_by_title in passes:
            if max_results and len(results) >= max_results:
                logger.info(f"Skipping {pass_name}: max_results ({max_results}) already reached.")
                break

            logger.info(f"Initiating {pass_name} for '{query}'...")
            page_num = 1
            total_pages: Optional[int] = None

            while page_num <= self.max_pages:
                if max_results and len(results) >= max_results:
                    break

                try:
                    model = await self._fetch_search_page(
                        query, page_num, search_by_title=search_by_title, pass_name=pass_name
                    )
                except Exception as e:
                    logger.error(f"[ERROR] {pass_name} page {page_num} failed after {MAX_RETRIES} attempts: {_short_error(e)}")
                    self.stats["pages_failed"] += 1
                    self.failed_pages.append(f"{pass_name} p{page_num}")
                    if total_pages is None:
                        # Without a successful first page we cannot know how many pages exist.
                        logger.error(f"Aborting {pass_name}: total page count is unknown.")
                        break
                    if page_num >= total_pages:
                        break
                    page_num += 1
                    continue

                items = model.get("collection") or []
                total_count = model.get("totalCount") or 0
                total_pages = model.get("totalPages") or 1
                self.stats["expected_records"] = max(self.stats["expected_records"], total_count)

                logger.info(f"[{pass_name}] Page {page_num}/{total_pages} | items={len(items)} | total matches={total_count}")

                if not items:
                    logger.info(f"{pass_name}: no items on page {page_num}, stopping this pass.")
                    break

                self.stats["pages_processed"] += 1
                self.stats["records_received"] += len(items)

                page_new = 0
                page_dupes = 0

                for item in items:
                    if max_results and len(results) >= max_results:
                        break

                    serial = (item.get("serial") or "").strip()
                    name = item.get("statuteName") or "Unknown"

                    if not serial:
                        logger.warning(f"Skipping malformed search item without serial: {name}")
                        self.stats["malformed_items"] += 1
                        continue

                    # Deduplication check
                    if serial in self._seen_serials:
                        self.stats["duplicates"] += 1
                        page_dupes += 1
                        continue
                    self._seen_serials.add(serial)
                    page_new += 1

                    logger.info(f"[{len(results)+1}/{max_results or total_count}] Fetching details: {name}")

                    # Fetch full detail including articles and metadata
                    await self._rate_limit()
                    detail_model = await self._fetch_statute_detail(serial)
                    if not detail_model:
                        self.stats["detail_failures"] += 1
                        self.failed_details.append(f"{serial} ({name})")

                    try:
                        legislation = parse_legislation_data(item, detail_model)
                    except Exception as e:
                        logger.error(f"Failed to parse legislation data for {name}: {e}")
                        self.stats["parse_failures"] += 1
                        continue

                    if not legislation.content:
                        self.stats["empty_content"] += 1
                        self.empty_content_titles.append(legislation.title or serial)
                        logger.warning(f"Legislation has EMPTY content: {legislation.title} ({serial})")

                    results.append(legislation)
                    self.stats["unique_legislations"] += 1

                logger.info(f"[{pass_name}] Page {page_num}/{total_pages} | new={page_new} | duplicates={page_dupes}")

                if page_num >= total_pages:
                    break

                page_num += 1
                await self._rate_limit()

        logger.info(
            f"Search complete. Unique: {self.stats['unique_legislations']}, "
            f"Duplicates: {self.stats['duplicates']}, "
            f"Detail failures: {self.stats['detail_failures']}, "
            f"Empty content: {self.stats['empty_content']}, "
            f"Pages processed: {self.stats['pages_processed']}, failed: {self.stats['pages_failed']}"
        )
        return results

    def save_results(
        self,
        results: List[Legislation],
        json_path: str = JSON_OUTPUT_FILE,
        csv_path: str = CSV_OUTPUT_FILE,
        db_path: str = DATABASE_PATH,
    ) -> None:
        """
        Save scraped results to JSON, CSV (this run only) and SQLite (cumulative).
        """
        if not results:
            logger.warning("No records to save.")
            return

        os.makedirs(OUTPUT_DIR, exist_ok=True)

        # 1. Save JSON (ensure_ascii=False for proper Arabic text)
        json_data = [r.to_dict() for r in results]
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(json_data, f, ensure_ascii=False, indent=2)
        self.output_locations["JSON"] = os.path.abspath(json_path)
        logger.info(f"Saved {len(results)} records to JSON: {os.path.abspath(json_path)}")

        # 2. Save CSV (utf-8-sig adds BOM for clean Arabic opening in Excel)
        with open(csv_path, "w", encoding="utf-8-sig", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
            writer.writeheader()
            for r in results:
                writer.writerow(r.to_csv_dict())
        self.output_locations["CSV"] = os.path.abspath(csv_path)
        logger.info(f"Saved {len(results)} records to CSV: {os.path.abspath(csv_path)}")

        # 3. Save to SQLite database
        try:
            create_tables(db_path)
            written = insert_legislations(results, db_path)
            self.output_locations["SQLite"] = os.path.abspath(db_path)
            logger.info(f"Saved {written} records to SQLite database: {os.path.abspath(db_path)}")
        except Exception as e:
            logger.error(f"Failed to save records to SQLite database: {e}")

    def print_summary(self, results: List[Legislation]) -> None:
        """
        Print execution summary and display an authentic sample record.
        """
        is_complete = self.stats["pages_failed"] == 0 and self.stats["detail_failures"] == 0

        print("\n" + "=" * 65)
        print("         MOJ LEGISLATION SCRAPER - RUN SUMMARY")
        print("=" * 65)
        if is_complete:
            print("  STATUS: FULL SCRAPE (SUCCESS)")
        else:
            print("  STATUS: INCOMPLETE SCRAPE")

        print(f"  Total matches reported:   {self.stats['expected_records']}")
        print(f"  Pages processed:          {self.stats['pages_processed']}")
        print(f"  Pages failed:             {self.stats['pages_failed']}")
        print(f"  Records received:         {self.stats['records_received']}")
        print(f"  Unique legislations:      {self.stats['unique_legislations']}")
        print(f"  Duplicates skipped:       {self.stats['duplicates']}")
        print(f"  Malformed items skipped:  {self.stats['malformed_items']}")
        print(f"  Detail failures:          {self.stats['detail_failures']}")
        print(f"  Parse failures:           {self.stats['parse_failures']}")
        print(f"  Empty-content records:    {self.stats['empty_content']}")
        for page in self.failed_pages:
            print(f"    - failed page: {page}")
        for detail in self.failed_details:
            print(f"    - failed detail: {detail}")
        for title in self.empty_content_titles:
            print(f"    - empty content: {title}")
        print("=" * 65)

        if results:
            print("\n  Records:")
            for r in results:
                print(f"  - {r.title} | serial={r.serial} | content={len(r.content or '')} chars")

            print("\n  Sample Scraped Record:")
            print("  " + "-" * 61)
            sample = results[0].to_dict()
            for key, val in sample.items():
                if key in ("content", "description") and val:
                    preview = str(val)[:200].replace("\n", " ")
                    print(f"  {key:20}: {preview}...")
                else:
                    print(f"  {key:20}: {val}")
            print("  " + "-" * 61)

        if self.output_locations:
            print("\n  Output locations:")
            for name, path in self.output_locations.items():
                print(f"  - {name:7}: {path}")
        print("=" * 65 + "\n")
