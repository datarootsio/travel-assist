"""What a retrieval function returns.

Scaffolding — ships implemented. Students write the *searches*, not the shape of
their results.

The first Pydantic model in the repo, and Pydantic for a reason: everything
before this was internal, and a dataclass is fine for a value that never leaves
the process. This one leaves — it reaches the model, in the prompt, and any
downstream caller of `hybrid_search`.

Two fields serve hybrid fusion rather than this milestone:

- `rank` — 1-based position in the ranking it came from. Reciprocal Rank Fusion
  works on ranks, not scores, because a cosine distance and a `ts_rank_cd` are
  not comparable quantities. Discarding the scores is what makes fusion honest.
- `method` — which search produced it, so a fused result can still say where
  each member came from.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

SearchMethod = Literal["vector", "keyword", "hybrid"]


class RetrievedChunk(BaseModel):
    """One passage returned by a search, with everything a citation needs.

    Attributes:
        chunk_id: Primary key in `chunks`. Stable within one build of the index,
            *not* across rebuilds — never persist it as a long-lived reference.
        title: Article title, e.g. `"Porto"`.
        url: Source URL. Present on every chunk because the corpus is CC BY-SA
            and attribution is part of the product.
        section_path: Breadcrumb such as `"Porto > Eat"`. `None` only if the
            article had no headings at all.
        content: The passage itself. This is the text an answer quotes; it does
            **not** include the section path, which is prepended for embedding
            only (see `ingestion/chunk.py`).
        token_count: Length of `content`, used by the context budget.
        score: The raw score from the search that produced this chunk. Comparable
            *within* one method and meaningless across methods — see the module
            docstring.
        rank: 1-based position within its own ranking.
        method: Which search produced it.
        country: Country the article belongs to, when resolved.
        continent: Continent the article belongs to, when resolved.
        article_type: `"city"`, `"park"`, `"region"`, ... when known.
    """

    chunk_id: int
    title: str
    url: str
    section_path: str | None = None
    content: str
    token_count: int = 0
    score: float
    rank: int = Field(ge=1)
    method: SearchMethod
    country: str | None = None
    continent: str | None = None
    article_type: str | None = None

    def citation_label(self) -> str:
        """A short human-readable source, for logs, tables and demo output."""
        return self.section_path or self.title


class SearchFilters(BaseModel):
    """Narrow a search to chunks whose document matches every field set here.

    A field left `None` places no constraint. Day 1 never sets any of these —
    `hybrid_search` retrieves once, unfiltered. They exist for a narrower,
    filtered retrieval once a caller knows which destination a question is
    actually about, e.g. after a first call surfaces both Lisbon and Porto.

    Attributes:
        country: Exact match against `RetrievedChunk.country`, e.g. `"Portugal"`.
        continent: Exact match against `RetrievedChunk.continent`, e.g. `"Europe"`.
        article_type: Exact match against `RetrievedChunk.article_type`, e.g. `"city"`.
    """

    country: str | None = None
    continent: str | None = None
    article_type: str | None = None
