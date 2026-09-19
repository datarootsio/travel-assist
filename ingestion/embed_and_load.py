"""Embed the selected corpus and load it into Postgres.

INSTRUCTOR ONLY. This is the long pole in the pre-course runbook — book time for
it well before the teaching day.

**Deviation from the original spec, deliberate.** It said the shipped corpus
includes embeddings. It ships text only, and vectors are computed at seed time by
whatever backend is configured, because shipped vectors would pin the corpus to
one embedding model: seeding it under a different one produces an index whose
metadata and contents disagree, which is the exact silent corruption the
frozen-embedding guard exists to prevent.
"""

from __future__ import annotations

import logging
import re
import time
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from langchain_core.embeddings import Embeddings
from psycopg import Connection

from ingestion.chunk import DEFAULT_OVERLAP_TOKENS, DEFAULT_TARGET_TOKENS, Chunk
from ingestion.corpus import CorpusArticle
from ingestion.embedding_cache import DEFAULT_CACHE_PATH, EmbeddingCache
from travel_assist.config import Settings, get_settings
from travel_assist.db import connect
from travel_assist.models import get_embeddings

logger = logging.getLogger(__name__)

# 100 is the maximum batch the embedding APIs accept, and bigger batches mean
# fewer round trips. Note that batching does **not** stretch a quota: providers
# meter embedding by the number of texts, not the number of HTTP requests, so a
# hundred chunks in one call costs exactly what a hundred calls would.
EMBED_BATCH_SIZE = 100
MAX_EMBED_ATTEMPTS = 6
BASE_RETRY_DELAY_SECONDS = 2.0
MAX_RETRY_DELAY_SECONDS = 120.0

_RATE_LIMIT_MARKERS = ("429", "rate limit", "quota", "too many requests")

# Providers usually say exactly how long to wait. Obeying that beats guessing.
_RETRY_HINT = re.compile(r"retry in (\d+(?:\.\d+)?)s", re.IGNORECASE)

Article = tuple[CorpusArticle, list[Chunk]]


def _is_rate_limit(error: Exception) -> bool:
    """Whether an exception looks like a rate limit rather than a real bug.

    Matched on the message rather than the type: every provider raises something
    different, and this file must keep working when the backend changes.
    """
    message = str(error).lower()
    return any(marker in message for marker in _RATE_LIMIT_MARKERS)


def _retry_delay(error: Exception, attempt: int, base_delay: float) -> float:
    """How long to wait before retrying a throttled batch.

    Prefers the provider's own `retry in Ns` hint over exponential backoff —
    guessing either wastes time or hammers an endpoint that already said no.
    """
    hint = _RETRY_HINT.search(str(error))
    if hint:
        return min(float(hint.group(1)) + 1.0, MAX_RETRY_DELAY_SECONDS)
    return min(base_delay * (2 ** (attempt - 1)), MAX_RETRY_DELAY_SECONDS)


@dataclass
class LoadStats:
    """What a load actually did, for the runbook's timing notes."""

    documents: int
    chunks: int
    seconds: float

    def describe(self) -> str:
        """One line suitable for printing at the end of a run."""
        rate = self.chunks / self.seconds if self.seconds else 0.0
        return (
            f"{self.documents} documents, {self.chunks} chunks in "
            f"{self.seconds:.1f}s ({rate:.0f} chunks/s)"
        )


def embed_all(
    texts: Sequence[str],
    *,
    settings: Settings | None = None,
    embeddings: Embeddings | None = None,
    cache: EmbeddingCache | None = None,
    batch_size: int = EMBED_BATCH_SIZE,
    max_attempts: int = MAX_EMBED_ATTEMPTS,
    base_delay: float = BASE_RETRY_DELAY_SECONDS,
) -> list[list[float]]:
    """Embed every chunk in batches, backing off when the endpoint pushes back.

    Batched because a request per chunk turns a ten-minute job into an hour, and
    retried because ingesting the real corpus means embedding tens of thousands
    of chunks against a metered endpoint. Being throttled part-way through is an
    expected event, not an exceptional one — losing an hour of embedding to a
    single 429 would be a miserable way to find that out.

    Retries are **logged, not silent** — the instructor watching the runbook
    should be able to see the endpoint throttling and understand why the run
    slowed down. Only rate-limit errors are retried; a malformed request fails
    immediately rather than burning quota.

    Args:
        texts: What to embed, in order.
        settings: Used to build the embedding client when one is not supplied.
        embeddings: An explicit client. Injected by tests.
        cache: Reuses vectors already paid for. Absent means embed everything.
        batch_size: Chunks per request.
        max_attempts: Attempts per batch before giving up.
        base_delay: Seconds for the first backoff; doubles each attempt.

    Returns:
        One vector per input text, in the same order.

    Raises:
        Exception: The last error, if a batch never succeeds.
    """
    if embeddings is None:
        embeddings = get_embeddings(settings=settings or get_settings())

    known: dict[str, list[float]] = cache.get_many(texts) if cache else {}

    # Deduplicate before paying. Wikivoyage boilerplate repeats across articles,
    # and embedding is metered per text, so the same string twice is money twice.
    outstanding = list(dict.fromkeys(text for text in texts if text not in known))

    if known:
        logger.info("reusing %d cached embeddings", len(texts) - len(outstanding))

    for start in range(0, len(outstanding), batch_size):
        batch = outstanding[start : start + batch_size]

        for attempt in range(1, max_attempts + 1):
            try:
                fresh = embeddings.embed_documents(list(batch))
                break
            except Exception as error:
                if not _is_rate_limit(error) or attempt == max_attempts:
                    raise
                delay = _retry_delay(error, attempt, base_delay)
                logger.warning(
                    "rate limited while embedding (attempt %d/%d); waiting %.0fs — %s",
                    attempt,
                    max_attempts,
                    delay,
                    str(error)[:120],
                )
                time.sleep(delay)

        batch_vectors = dict(zip(batch, fresh, strict=True))
        known.update(batch_vectors)

        # Persist per batch, not at the end: a run that dies half way through
        # should not throw away the quota it already spent.
        if cache:
            cache.put_many(batch_vectors)

        logger.info(
            "embedded %d/%d new chunks", min(start + batch_size, len(outstanding)), len(outstanding)
        )

    return [known[text] for text in texts]


def apply_chunk_budget(articles: Sequence[Article], max_chunks: int | None) -> list[Article]:
    """Trim the corpus so it costs no more than `max_chunks` embeddings.

    Embedding is metered per chunk and free tiers are capped per day, so an
    ingestion that overruns its budget does not merely cost money — it stops
    partway and cannot be resumed until tomorrow. This bounds the spend before
    a single request is made.

    Articles are dropped **whole**. A half-loaded article is worse than an absent
    one: its citations would point at passages the index does not contain.

    Args:
        articles: Chunked articles, in selection order.
        max_chunks: Hard upper bound, or None for no limit.

    Returns:
        The longest prefix of `articles` that fits.
    """
    if max_chunks is None:
        return list(articles)

    kept: list[Article] = []
    total = 0

    for candidate, chunks in articles:
        if total + len(chunks) > max_chunks:
            logger.warning(
                "chunk budget %d reached — dropping %r (%d chunks) and everything after it",
                max_chunks,
                candidate.title,
                len(chunks),
            )
            break
        kept.append((candidate, chunks))
        total += len(chunks)

    return kept


def _embed_corpus(
    texts: Sequence[str],
    *,
    settings: Settings,
    embeddings: Embeddings | None,
) -> list[list[float]]:
    """Embed a whole corpus, caching only what a real provider was paid for.

    The cache exists to stop a re-run buying the same vectors twice. An injected
    client is a test double, so its vectors are deliberately kept out of it:
    caching them would mean a later real run silently reading them back as
    though they came from the provider.
    """
    if embeddings is not None:
        return embed_all(texts, embeddings=embeddings)

    with EmbeddingCache(
        DEFAULT_CACHE_PATH,
        model=settings.embedding_model_name,
        dimension=settings.embedding_dim,
    ) as cache:
        return embed_all(texts, settings=settings, cache=cache)


def _write_index_metadata(
    conn: Connection[Any],
    settings: Settings,
    *,
    dump_date: str,
    n_docs: int,
    n_chunks: int,
) -> None:
    """Record the contract the startup guard checks on every run."""
    rows = {
        "embedding_model": settings.embedding_model_name,
        "embedding_dim": str(settings.embedding_dim),
        "dump_date": dump_date,
        # Derived, never written out by hand. This string is how someone later
        # explains why a corpus retrieves the way it does, and a hardcoded one
        # went stale the first time the chunker moved.
        "chunk_strategy": (
            f"section-split, packed to {DEFAULT_TARGET_TOKENS} tokens "
            f"with {DEFAULT_OVERLAP_TOKENS} overlap"
        ),
        "built_at": datetime.now(UTC).isoformat(),
        "n_docs": str(n_docs),
        "n_chunks": str(n_chunks),
    }

    with conn.cursor() as cursor:
        for key, value in rows.items():
            cursor.execute(
                "INSERT INTO index_metadata (key, value) VALUES (%s, %s) "
                "ON CONFLICT (key) DO UPDATE SET value = EXCLUDED.value",
                (key, value),
            )


def load_articles(
    articles: Iterable[Article],
    *,
    settings: Settings | None = None,
    dump_date: str,
    embeddings: Embeddings | None = None,
) -> LoadStats:
    """Embed and insert articles, replacing any existing corpus.

    Re-runnable on purpose: a second run must produce the same index, not a
    doubled one. Documents are upserted on `page_id` and their chunks replaced.

    Args:
        articles: Pairs of selected article and its chunks.
        settings: Configuration; defaults to the process-wide Settings.
        dump_date: The Wikivoyage dump date, recorded in `index_metadata` and
            in the manifest so the corpus is reproducible.
        embeddings: An explicit client, mirroring `embed_all`. Injected by tests
            so the database behaviour here — idempotent re-runs, orphan removal,
            the recorded contract — stays covered with no credentials and no
            quota. A real run leaves this unset.

    Returns:
        Counts and elapsed time, for the runbook's timings.
    """
    settings = settings or get_settings()
    started = time.monotonic()
    materialised = list(articles)

    texts = [chunk.embedding_text for _, chunks in materialised for chunk in chunks]
    if not texts:
        return LoadStats(documents=0, chunks=0, seconds=0.0)

    vectors = _embed_corpus(texts, settings=settings, embeddings=embeddings)

    inserted_chunks = 0
    position = 0

    with connect(settings) as conn:
        with conn.cursor() as cursor:
            for candidate, chunks in materialised:
                cursor.execute(
                    "INSERT INTO documents "
                    "(page_id, title, url, country, continent, article_type, status, "
                    " lat, lon, revision_id, retrieved_at) "
                    "VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s, now()) "
                    "ON CONFLICT (page_id) DO UPDATE SET "
                    "  title = EXCLUDED.title, url = EXCLUDED.url, "
                    "  country = EXCLUDED.country, continent = EXCLUDED.continent, "
                    "  article_type = EXCLUDED.article_type, status = EXCLUDED.status, "
                    "  lat = EXCLUDED.lat, lon = EXCLUDED.lon, "
                    "  revision_id = EXCLUDED.revision_id, retrieved_at = now() "
                    "RETURNING id",
                    (
                        candidate.page_id,
                        candidate.title,
                        candidate.url,
                        candidate.country,
                        candidate.continent,
                        candidate.article_type,
                        candidate.status,
                        candidate.lat,
                        candidate.lon,
                        candidate.revision_id,
                    ),
                )
                row = cursor.fetchone()
                if row is None:  # pragma: no cover - RETURNING always yields a row
                    raise RuntimeError(f"insert of {candidate.title!r} returned no id")
                document_id = row["id"]

                # Replace rather than append, so a re-run is idempotent.
                cursor.execute("DELETE FROM chunks WHERE document_id = %s", (document_id,))

                for chunk in chunks:
                    cursor.execute(
                        "INSERT INTO chunks "
                        "(document_id, chunk_index, section_path, content, token_count, "
                        " embedding) VALUES (%s,%s,%s,%s,%s,%s)",
                        (
                            document_id,
                            chunk.chunk_index,
                            chunk.section_path,
                            chunk.content,
                            chunk.token_count,
                            str(vectors[position]),
                        ),
                    )
                    position += 1
                    inserted_chunks += 1

            # Drop anything the previous run loaded that is not in this one.
            # Without this, a re-run with a changed selection leaves orphaned
            # documents behind: `index_metadata` then reports what this run
            # loaded while the tables hold strictly more, so the corpus size is
            # quietly wrong and dropped destinations stay retrievable.
            cursor.execute(
                "DELETE FROM documents WHERE NOT (page_id = ANY(%s))",
                ([candidate.page_id for candidate, _ in materialised],),
            )
            if cursor.rowcount:
                logger.info("removed %d documents no longer in the corpus", cursor.rowcount)

            _write_index_metadata(
                conn,
                settings,
                dump_date=dump_date,
                n_docs=len(materialised),
                n_chunks=inserted_chunks,
            )
        conn.commit()

    return LoadStats(
        documents=len(materialised),
        chunks=inserted_chunks,
        seconds=time.monotonic() - started,
    )
