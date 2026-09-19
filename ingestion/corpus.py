"""Read `data/corpus.jsonl` — the shipped, pre-cleaned corpus — and write its manifest.

Scaffolding, ships implemented. This is the easiest boundary in the corpus
to get wrong: the file holds
**articles, not chunks, and no vectors**. Chunks would freeze chunk size into
the shipped file; vectors would pin it to one embedding model and reintroduce
the silent corruption the frozen-embedding guard exists to catch. Both are
computed at seed time, by `ingestion/run.py`, so chunk size and embedding model
stay free parameters forever.

`data/corpus.jsonl` is produced once by `scripts/corpus_build/build_corpus.py`
and never touched by hand — nothing here writes it back.
"""

from __future__ import annotations

import csv
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

from ingestion.chunk import Section


@dataclass(frozen=True)
class CorpusArticle:
    """One cleaned Wikivoyage article, as stored in `data/corpus.jsonl`.

    Attributes:
        sections: Already split and markup-free. Chunking happens
            later, in `ingestion/chunk.py`, from these.
    """

    page_id: int
    title: str
    url: str
    country: str | None
    continent: str | None
    article_type: str
    status: str
    lat: float | None
    lon: float | None
    revision_id: int
    dump_date: str
    sections: list[Section] = field(default_factory=list)


def read_corpus(path: Path) -> list[CorpusArticle]:
    """Load every article from `data/corpus.jsonl`, in file order."""
    articles: list[CorpusArticle] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            sections = [Section(path=s["path"], text=s["text"]) for s in record["sections"]]
            articles.append(CorpusArticle(**{**record, "sections": sections}))
    return articles


def write_manifest(
    articles: Sequence[CorpusArticle],
    path: Path,
    *,
    chunk_counts: Mapping[int, int] | None = None,
) -> None:
    """Write the published "which cities are in there" list.

    Committed to the repo on purpose. Wikivoyage is CC BY-SA, so the licence and
    the dump date travel with the corpus, and `revision_id` makes the whole
    thing reproducible. Regenerated from `data/corpus.jsonl` — the dump is never
    needed to answer "which cities are in there?".
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    dump_date = articles[0].dump_date if articles else "unknown"

    with path.open("w", newline="", encoding="utf-8") as handle:
        handle.write(
            f"# Wikivoyage corpus for travel-assist. Source: English Wikivoyage dump "
            f"{dump_date}.\n"
            "# Content is CC BY-SA (dual-licensed 3.0 / 4.0). Every chunk keeps its source "
            "URL and every answer cites it.\n"
            "# Generated from data/corpus.jsonl by ingestion/run.py — do not hand-edit.\n"
        )
        writer = csv.writer(handle)
        writer.writerow(
            [
                "page_id",
                "title",
                "url",
                "country",
                "continent",
                "article_type",
                "status",
                "n_chunks",
                "revision_id",
            ]
        )
        for article in articles:
            writer.writerow(
                [
                    article.page_id,
                    article.title,
                    article.url,
                    article.country or "",
                    article.continent or "",
                    article.article_type,
                    article.status,
                    (chunk_counts or {}).get(article.page_id, ""),
                    article.revision_id,
                ]
            )
