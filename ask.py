"""
Ask a legal question from the command line.

Usage:
    python ask.py "ما هي شروط افتتاح إجراء التسوية الوقائية؟"
    python ask.py "..." --mode rag        # plain RAG instead of the agent
"""

import argparse
import logging
import sys

from config import LOG_FORMAT


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

    parser = argparse.ArgumentParser(description="Ask a question about the scraped Saudi legislation")
    parser.add_argument("question")
    parser.add_argument("--mode", choices=["agent", "rag"], default="agent")
    parser.add_argument("--log-level", default="WARNING")
    args = parser.parse_args()
    logging.basicConfig(level=args.log_level.upper(), format=LOG_FORMAT, stream=sys.stdout)

    from rag.rag import RAGPipeline
    pipeline = RAGPipeline()
    if args.mode == "agent":
        from rag.agent import LegalAgent
        result = LegalAgent(pipeline).ask(args.question)
    else:
        result = pipeline.answer(args.question)

    print("=" * 70)
    print(f"Mode: {result['mode']}")
    print("=" * 70)
    print(result["answer"])
    print("\nSources:")
    for s in result["sources"]:
        refs = " ".join(f"[{c['ref']}]" for c in s["chunks"])
        print(f"  - {s['title']} {refs}\n    {s['source_url']}")
    for step in result.get("steps", []):
        print(f"  step: {step['tool']} {step['input']} -> {step['result']}")
    print(f"\n{result['disclaimer']}")


if __name__ == "__main__":
    main()
