"""
MOJ Laws & Regulations Web Scraper - Main Entry Point

Usage:
    python main.py --query "الإفلاس"
    python main.py --query "الإفلاس" --max-results 5
    python main.py --query "العمل" --delay 2.0
    python main.py --query "التجارة" --no-headless
"""

import argparse
import asyncio
import logging
import os
import sys

# Ensure UTF-8 console output and environment on Windows
os.environ["PYTHONIOENCODING"] = "utf-8"
if sys.platform == "win32":
    try:
        import ctypes
        kernel32 = ctypes.windll.kernel32
        kernel32.SetConsoleOutputCP(65001)
        kernel32.SetConsoleCP(65001)
    except Exception:
        os.system("chcp 65001 > nul 2>&1")

    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

from config import LOG_FORMAT, LOG_LEVEL, REQUEST_DELAY_SECONDS
from scraper import MOJScraper


def setup_logging(level_name: str = LOG_LEVEL) -> None:
    """Configure structured logging with UTF-8 support."""
    handler = logging.StreamHandler(sys.stdout)
    handler.setStream(sys.stdout)
    logging.basicConfig(
        level=getattr(logging, level_name.upper(), logging.INFO),
        format=LOG_FORMAT,
        handlers=[handler],
    )
    # Silence noisy third-party loggers
    logging.getLogger("playwright").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("asyncio").setLevel(logging.WARNING)


async def run_scraper(args: argparse.Namespace) -> None:
    """Run scraper lifecycle."""
    logger = logging.getLogger(__name__)

    logger.info("Initializing Saudi MOJ Legislation Scraper...")
    logger.info(f"Target query: '{args.query}'")
    logger.info(f"Max results limit: {args.max_results or 'Unlimited'}")
    logger.info(f"Headless mode: {not args.no_headless}")
    logger.info(f"Polite delay: {args.delay}s")

    async with MOJScraper(
        headless=not args.no_headless,
        delay=args.delay,
    ) as scraper:
        results = await scraper.search(
            query=args.query,
            max_results=args.max_results,
        )

        scraper.save_results(results)
        scraper.print_summary(results)


def main():
    """CLI Argument Parsing and Runner."""
    parser = argparse.ArgumentParser(
        description="Official Saudi Ministry of Justice (MOJ) Legislation Scraper",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --query "الإفلاس"
  python main.py --query "الإفلاس" --max-results 5
  python main.py --query "المحاكم التجارية" --max-results 10
  python main.py --query "التنفيذ" --delay 2.5
        """,
    )

    parser.add_argument(
        "--query", "-q",
        type=str,
        required=True,
        help="Search query in Arabic (e.g., 'الإفلاس' or 'العمل')",
    )
    parser.add_argument(
        "--max-results", "-m",
        type=int,
        default=None,
        help="Maximum number of legislation records to extract (default: all matching)",
    )
    parser.add_argument(
        "--no-headless",
        action="store_true",
        default=False,
        help="Run browser in visible mode (default: headless)",
    )
    parser.add_argument(
        "--delay", "-d",
        type=float,
        default=REQUEST_DELAY_SECONDS,
        help=f"Polite delay between requests in seconds (default: {REQUEST_DELAY_SECONDS}s)",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default=LOG_LEVEL,
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help=f"Logging verbosity level (default: {LOG_LEVEL})",
    )

    args = parser.parse_args()
    setup_logging(args.log_level)

    try:
        asyncio.run(run_scraper(args))
    except KeyboardInterrupt:
        print("\nScraper interrupted by user.")
        sys.exit(0)
    except Exception as e:
        logging.getLogger(__name__).critical(f"Fatal scraper error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
