"""`RetrievedChunk` — the first Pydantic boundary in the repo.

It is Pydantic rather than a dataclass because on day 2 this object comes back
out of a tool call, which means the model sees it. That the required fields are
required and the enum rejects an unknown value is Pydantic's behaviour, not
ours, and is not tested here.

What *is* ours is the citation label: which of two fields a citation shows, and
what it falls back to when the better one is missing. Every cited answer in the
course renders through it.
"""

from __future__ import annotations

from travel_assist.retrieval.models import RetrievedChunk


def chunk(**overrides: object) -> RetrievedChunk:
    fields = {
        "chunk_id": 1,
        "title": "Porto",
        "url": "https://en.wikivoyage.org/wiki/Porto",
        "section_path": "Porto > Eat",
        "content": "The francesinha is a sandwich.",
        "token_count": 7,
        "score": 0.82,
        "rank": 1,
        "method": "vector",
    }
    fields.update(overrides)
    return RetrievedChunk(**fields)  # type: ignore[arg-type]


def test_the_citation_label_prefers_the_section_path():
    """A section path tells a reader where in the article the claim came from."""
    assert chunk().citation_label() == "Porto > Eat"


def test_the_citation_label_falls_back_to_the_title():
    """`section_path` is frequently unresolved; a citation is still required."""
    assert chunk(section_path=None).citation_label() == "Porto"
