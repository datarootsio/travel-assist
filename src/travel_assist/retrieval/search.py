"""Searching the index: dense, sparse, and — once hybrid lands — the two fused.

Diff from the previous milestone: the index that was built is now queried.

Read this file top to bottom. It is deliberately one module rather than three,
because the lesson here *is* the comparison: two functions, nearly identical
in shape, differing only in how they decide what "relevant" means, and each
failing exactly where the other succeeds.

    vector_search    embeddings      finds meaning, misses rare words
    keyword_search   tsvector        finds words, misses paraphrase
    hybrid_search    RRF             fuses both rankings

Diff from vector-search/keyword-search: `reciprocal_rank_fusion` and
`hybrid_search` land. `hybrid_search` is travel-assist's frozen retrieval
entrypoint — its signature is frozen from here, called directly by day 1's
pipeline and available unmodified to any downstream consumer.

Three decisions worth the minute they take:

**The vector operator must match the index.** `002_indexes.sql` builds HNSW with
`vector_cosine_ops`, so `vector_search` queries with `<=>`. Use `<->` (L2) and you
still get results, in a plausible order, with no error — Postgres simply cannot
use an index built for a different operator and scans every row instead.
Imperceptible on this corpus, seconds instead of milliseconds at 100k chunks.
Worth showing live with `EXPLAIN`.

**`websearch_to_tsquery`, not `plainto_tsquery`.** `plainto_tsquery` ANDs every
word and understands nothing else, so quotes and a leading `-` are silently
dropped. `websearch_to_tsquery` implements the syntax people already know from
every search box, and — the part that matters here — never raises on malformed
input. `to_tsquery` does, and this takes raw text from a chat box.

**Both return `score` as "higher is better".** pgvector gives cosine *distance*,
where smaller is better, so it is flipped to `1 - distance`. Two rankings that are
about to be compared should not disagree about which direction is good.

`tsv` is `GENERATED ALWAYS` in the schema, so both searches provably see the same
text. That is what makes the bake-off fair rather than accidental.
"""

from __future__ import annotations

from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor

from langchain_core.embeddings import Embeddings
from psycopg.rows import dict_row

from travel_assist.config import Settings, get_settings
from travel_assist.db import connect
from travel_assist.models import get_embeddings
from travel_assist.retrieval.models import RetrievedChunk, SearchFilters

# Every chunk column a citation or the context budget needs, plus its document.
# Shared so the two searches provably return the same shape — the only thing
# that may differ between them is the ranking.
_SELECT = """
    SELECT c.id AS chunk_id,
           d.title, d.url, d.country, d.continent, d.article_type,
           c.section_path, c.content, c.token_count,
"""

_FROM = """
    FROM chunks c
    JOIN documents d ON d.id = c.document_id
"""


def vector_search(
    query: str,
    k: int = 8,
    *,
    settings: Settings | None = None,
    embeddings: Embeddings | None = None,
) -> list[RetrievedChunk]:
    """Retrieve the k passages whose embeddings are nearest to `query`'s.

    Args:
        query: Raw user text, embedded with the model the index was built with.
            A mismatch degrades retrieval silently rather than failing, which is
            what the frozen-embedding guard exists to catch.
        k: Maximum results.
        settings: Configuration; defaults to the process-wide Settings.
        embeddings: An explicit embedding client. Injected by tests; built from
            `settings` otherwise.

    Returns:
        Up to `k` chunks ordered by descending cosine similarity, ranked from 1.
        An empty or whitespace-only query returns an empty list without embedding
        anything — embedding is metered, and an empty query has no useful nearest
        neighbour.
    """
    settings = settings or get_settings()

    if not query.strip():
        return []

    embeddings = embeddings or get_embeddings(settings=settings)
    query_vector = str(embeddings.embed_query(query))

    with connect(settings) as conn, conn.cursor(row_factory=dict_row) as cursor:
        cursor.execute(
            _SELECT
            + "1 - (c.embedding <=> %(query)s::vector) AS score"
            + _FROM
            # Order by the operator, not by the `score` alias. Both give the same
            # rows in the same order; only this one can use the HNSW index.
            + """
            ORDER BY c.embedding <=> %(query)s::vector, c.id
            LIMIT %(k)s
            """,
            {"query": query_vector, "k": k},
        )
        rows = cursor.fetchall()

    return [
        RetrievedChunk(method="vector", rank=rank, **row) for rank, row in enumerate(rows, start=1)
    ]


def keyword_search(
    query: str,
    k: int = 8,
    *,
    settings: Settings | None = None,
) -> list[RetrievedChunk]:
    """Retrieve the k passages whose text best matches `query`'s terms.

    Ranked with `ts_rank_cd` — cover density, which rewards matches appearing
    close together. That is most of the difference between a passage about the
    Rijksmuseum and one that mentions it in passing on the way somewhere else.

    Args:
        query: Raw user text. Web-search syntax works: quoted phrases, `or`, and
            a leading `-` to exclude.
        k: Maximum results.
        settings: Configuration; defaults to the process-wide Settings.

    Returns:
        Up to `k` chunks ordered by descending `ts_rank_cd`, ranked from 1. A
        query with no searchable terms — empty, punctuation only, or nothing but
        stop words — returns an empty list rather than raising.
    """
    settings = settings or get_settings()

    if not query.strip():
        return []

    with connect(settings) as conn, conn.cursor(row_factory=dict_row) as cursor:
        cursor.execute(
            _SELECT
            + "ts_rank_cd(c.tsv, websearch_to_tsquery('english', %(query)s)) AS score"
            + _FROM
            # `@@` is what uses the GIN index. The ranking function does not, so
            # the match has to be a WHERE clause, not just an ORDER BY.
            + """
            WHERE c.tsv @@ websearch_to_tsquery('english', %(query)s)
            ORDER BY score DESC, c.id
            LIMIT %(k)s
            """,
            {"query": query, "k": k},
        )
        rows = cursor.fetchall()

    return [
        RetrievedChunk(method="keyword", rank=rank, **row) for rank, row in enumerate(rows, start=1)
    ]


def reciprocal_rank_fusion(
    rankings: Sequence[Sequence[RetrievedChunk]], k_rrf: int = 60
) -> list[RetrievedChunk]:
    """Fuse two or more rankings of the same chunks into one, by hand.

    Every chunk's fused score is the sum, across every ranking it appears in, of
    `1 / (k_rrf + rank)` — its 1-based position in that ranking. A chunk near the
    top of either ranking scores well; a chunk near the top of *both* scores
    higher still, without ever comparing a cosine similarity to a `ts_rank_cd`
    value — they are not the same kind of number, which is exactly why fusion
    works on ranks rather than raw scores.

    Args:
        rankings: One ranking per method, each already ordered best-first with
            `rank` set from 1 (what `vector_search` and `keyword_search`
            return). A chunk absent from a ranking simply contributes nothing
            from it — this is how a method that found nothing still lets the
            other one win.
        k_rrf: Dampens how much rank 1 dominates rank 30. 60 is the constant
            from the original paper; it is stable enough that tuning it on ten
            queries would just be fitting the golden set, not improving fusion.

    Returns:
        Every distinct chunk (by `chunk_id`) across all rankings, ordered by
        descending fused score, re-ranked from 1, with `method="hybrid"`. `score`
        is the fused value — not comparable to either input ranking's `score`.
    """
    raise NotImplementedError("rrf — see the docstring above")


def _matches(chunk: RetrievedChunk, filters: SearchFilters) -> bool:
    """True if `chunk`'s document satisfies every field `filters` sets."""
    return (
        (filters.country is None or chunk.country == filters.country)
        and (filters.continent is None or chunk.continent == filters.continent)
        and (filters.article_type is None or chunk.article_type == filters.article_type)
    )


def hybrid_search(
    query: str,
    k: int = 8,
    *,
    vector_k: int = 30,
    keyword_k: int = 30,
    filters: SearchFilters | None = None,
    settings: Settings | None = None,
    embeddings: Embeddings | None = None,
) -> list[RetrievedChunk]:
    """Retrieve the k most relevant Wikivoyage chunks for `query`.

    Runs dense (pgvector) and sparse (tsvector) retrieval in parallel and fuses
    the two rankings with Reciprocal Rank Fusion.

    This function is the seam between the GenAI and Agentic AI days: tomorrow it
    is wrapped as an agent tool WITHOUT modification. Do not change its
    signature — tests/test_seam.py enforces it.

    Args:
        query: Raw user text, forwarded to both underlying searches unchanged.
        k: Maximum fused results returned.
        vector_k: Candidates dense retrieval contributes to the fusion.
        keyword_k: Candidates sparse retrieval contributes to the fusion.
        filters: Keep only chunks whose document matches every field set here.
            Applied to each ranking after retrieval and before fusion, so a
            narrow filter combined with a small vector_k/keyword_k can
            under-retrieve — widen either when filtering aggressively.
        settings: Configuration; defaults to the process-wide Settings.
        embeddings: An explicit embedding client, forwarded to `vector_search`.
            Injected by tests; built from `settings` otherwise.

    Returns:
        Up to `k` chunks, ordered by descending Reciprocal Rank Fusion score,
        ranked from 1, `method="hybrid"`.
    """
    raise NotImplementedError("hybrid-search — see the docstring above")
