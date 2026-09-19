"""Chunk, embed and load the shipped corpus. INSTRUCTOR ONLY.

    uv run python -m ingestion.run                  # the whole corpus
    uv run python -m ingestion.run --max-chunks 500  # cap embedding spend

Reads `data/corpus.jsonl` — never the dump. That split exists so chunk size and
embedding model stay free parameters: re-run this whenever either changes,
without ever touching `scripts/corpus_build/` again. See `ingestion/README.md`.
"""

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

from ingestion.chunk import DEFAULT_OVERLAP_TOKENS, DEFAULT_TARGET_TOKENS, chunk_article
from ingestion.corpus import read_corpus, write_manifest
from ingestion.embed_and_load import Article, apply_chunk_budget, load_articles
from travel_assist.config import get_settings

logger = logging.getLogger(__name__)

CORPUS_PATH = Path("data/corpus.jsonl")
MANIFEST_PATH = Path("data/corpus_manifest.csv")


def main() -> None:
    """Chunk every article in `data/corpus.jsonl`, embed it and load Postgres."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, default=CORPUS_PATH)
    parser.add_argument("--target-tokens", type=int, default=DEFAULT_TARGET_TOKENS)
    parser.add_argument("--overlap-tokens", type=int, default=DEFAULT_OVERLAP_TOKENS)
    parser.add_argument(
        "--max-chunks",
        type=int,
        default=None,
        help=(
            "hard cap on embedded chunks. Embedding is metered per chunk and free tiers "
            "reset daily, so overrunning stops the run until tomorrow."
        ),
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    settings = get_settings()

    if not args.corpus.exists():
        raise SystemExit(
            f"{args.corpus} not found — build it once with "
            "`uv run python -m scripts.corpus_build.build_corpus`, see ingestion/README.md"
        )

    logger.info("%s · corpus %s", settings.describe_backends(), args.corpus)

    started = time.monotonic()
    corpus_articles = read_corpus(args.corpus)
    logger.info("read %d articles from %s", len(corpus_articles), args.corpus)

    articles: list[Article] = [
        (
            article,
            chunk_article(
                article.sections,
                target_tokens=args.target_tokens,
                overlap_tokens=args.overlap_tokens,
            ),
        )
        for article in corpus_articles
    ]
    logger.info(
        "chunked %d articles into %d chunks",
        len(articles),
        sum(len(chunks) for _, chunks in articles),
    )

    articles = apply_chunk_budget(articles, args.max_chunks)
    if args.max_chunks:
        logger.info(
            "after budget: %d articles, %d chunks",
            len(articles),
            sum(len(chunks) for _, chunks in articles),
        )

    # Written after chunking so the manifest can report real chunk counts —
    # "which cities, and how much of each" is more useful than just the list.
    selected = [article for article, _ in articles]
    chunk_counts = {article.page_id: len(chunks) for article, chunks in articles}
    write_manifest(selected, MANIFEST_PATH, chunk_counts=chunk_counts)

    dump_date = corpus_articles[0].dump_date if corpus_articles else "unknown"
    stats = load_articles(articles, settings=settings, dump_date=dump_date)
    logger.info("loaded: %s", stats.describe())
    logger.info("total elapsed %.0fs", time.monotonic() - started)


if __name__ == "__main__":
    main()
