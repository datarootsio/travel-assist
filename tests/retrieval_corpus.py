"""A tiny, fully controlled corpus for retrieval tests.

Ten passages across three cities, each with a **hand-written embedding**.

The text is real, so Postgres stemming and `websearch_to_tsquery` parsing are
genuinely exercised. The vectors are hand-placed on orthogonal concept axes, so
"this query is nearer to that passage" is a fact of the fixture rather than a
property of whichever embedding model happens to be configured.

**What this fixture does not prove.** The scripted proper-noun miss *models* the
real failure but does not demonstrate it — the geometry was chosen by hand. The
demonstration is `travel-assist search` against the real index. This fixture
only stops the behaviour regressing once hybrid search fuses the two.

It is modelled on a query that really does fail: a street address inside a
hospital listing. An earlier version used "Rijksmuseum", which turned out to be a
*bad* example — on the real corpus both methods find it, because the passage
containing the word is also the passage the topic points at. The trick is a
proper noun whose surrounding text is about something else.
"""

from __future__ import annotations

import math

import psycopg
from psycopg.rows import dict_row

from travel_assist.config import Settings
from travel_assist.db import dsn

from .conftest import TEST_EMBEDDING_MODEL

EMBEDDING_DIM = 1536

# Concept axes. Orthogonal, so a weight is a similarity and nothing interferes.
AXES = {"water": 0, "art": 1, "food": 2, "transport": 3, "sights": 4, "health": 5}


def vector(**weights: float) -> list[float]:
    """A unit vector with the given weight on each named axis."""
    values = [0.0] * EMBEDDING_DIM
    for axis, weight in weights.items():
        values[AXES[axis]] = weight

    norm = math.sqrt(sum(value * value for value in values))
    if norm == 0:  # pragma: no cover - a zero query vector is never useful
        raise ValueError("a zero vector has no direction and cannot be compared")
    return [value / norm for value in values]


DOCUMENTS = [
    {
        "page_id": 1036,
        "title": "Amsterdam",
        "url": "https://en.wikivoyage.org/wiki/Amsterdam",
        "country": "Netherlands",
        "continent": "Europe",
        "article_type": "city",
        "status": "guide",
    },
    {
        "page_id": 28205,
        "title": "Porto",
        "url": "https://en.wikivoyage.org/wiki/Porto",
        "country": "Portugal",
        "continent": "Europe",
        "article_type": "city",
        "status": "guide",
    },
    {
        "page_id": 18579,
        "title": "Kyoto",
        "url": "https://en.wikivoyage.org/wiki/Kyoto",
        "country": "Japan",
        "continent": "Asia",
        "article_type": "city",
        "status": "guide",
    },
]

# (page_id, section_path, content, embedding)
CHUNKS = [
    (
        1036,
        "Amsterdam > See",
        "The Rijksmuseum holds Rembrandt's Night Watch and opens daily at 09:00.",
        vector(art=1.0),
    ),
    (
        1036,
        "Amsterdam > Do",
        "The lido at Sloterplas offers open water bathing throughout the summer.",
        vector(water=1.0),
    ),
    (
        1036,
        "Amsterdam > Eat",
        "Herring stalls sell broodje haring on bridges across the old centre.",
        vector(food=1.0),
    ),
    (
        1036,
        "Amsterdam > Understand",
        "The city counts more than fifty museums, several of them world class.",
        vector(art=0.9, sights=0.44),
    ),
    (
        28205,
        "Porto > See",
        "Livraria Lello is a neo-gothic bookshop with a crimson staircase.",
        vector(art=0.5, sights=0.87),
    ),
    (
        28205,
        "Porto > Do",
        "You can take a dip at Praia do Ourigo, a sheltered city beach.",
        vector(water=0.95, sights=0.31),
    ),
    (
        28205,
        "Porto > Eat",
        "The francesinha is a sandwich buried under melted cheese and beer sauce.",
        vector(food=0.9, sights=0.44),
    ),
    (
        18579,
        "Kyoto > Get around",
        "The Shinkansen stops at Kyoto Station on the Tokaido line.",
        vector(transport=1.0),
    ),
    (
        18579,
        "Kyoto > See",
        "Fushimi Inari is famous for its thousands of vermilion torii gates.",
        vector(sights=1.0),
    ),
    # The proper-noun case, modelled on a real one. A street address inside a
    # hospital listing: the rare token carries the meaning, and the passage
    # around it is about emergency care, so a dense embedding of the address
    # alone lands nowhere near it.
    (
        1036,
        "Amsterdam > Stay safe > Hospitals",
        "OLVG West at Jan Tooropstraat 164 has a 24-hour emergency department.",
        # Its own axis: emergency care is not "a thing to see", and the point of
        # the example is that the address embeds nowhere near the sights.
        vector(health=1.0),
    ),
]

# What each test query embeds to. Placed by hand — see the module docstring.
QUERY_VECTORS = {
    # A semantic query: near the bathing passages, though it shares no word with
    # either of them.
    "somewhere to swim outdoors": vector(water=1.0),
    # A rare proper noun: the embedding lands among *other sights*, not on the
    # passage that actually contains the word. This is the paired failure, and it
    # is what the real index does too: searching a Marrakech doctor's name
    # returns a photography museum, a mosque and a restaurant.
    "Jan Tooropstraat": vector(sights=1.0),
    "where can I eat": vector(food=1.0),
    "getting around by train": vector(transport=1.0),
}


def seed_corpus(settings: Settings) -> None:
    """Load the fixture corpus into an already-empty index."""
    with psycopg.connect(dsn(settings), row_factory=dict_row) as conn:
        with conn.cursor() as cursor:
            ids: dict[int, int] = {}
            for document in DOCUMENTS:
                cursor.execute(
                    "INSERT INTO documents (page_id, title, url, country, continent, "
                    " article_type, status, revision_id, retrieved_at) "
                    "VALUES (%(page_id)s, %(title)s, %(url)s, %(country)s, %(continent)s, "
                    " %(article_type)s, %(status)s, 1, now()) RETURNING id",
                    document,
                )
                row = cursor.fetchone()
                assert row is not None
                ids[int(document["page_id"])] = row["id"]

            for index, (page_id, section_path, content, embedding) in enumerate(CHUNKS):
                cursor.execute(
                    "INSERT INTO chunks (document_id, chunk_index, section_path, content, "
                    " token_count, embedding) VALUES (%s, %s, %s, %s, %s, %s)",
                    (
                        ids[page_id],
                        index,
                        section_path,
                        content,
                        len(content.split()),
                        str(embedding),
                    ),
                )

            for key, value in {
                "embedding_model": TEST_EMBEDDING_MODEL,
                "embedding_dim": str(EMBEDDING_DIM),
                "dump_date": "2026-09-01",
                "chunk_strategy": "fixture",
                "n_docs": str(len(DOCUMENTS)),
                "n_chunks": str(len(CHUNKS)),
            }.items():
                cursor.execute(
                    "INSERT INTO index_metadata (key, value) VALUES (%s, %s) "
                    "ON CONFLICT (key) DO UPDATE SET value = EXCLUDED.value",
                    (key, value),
                )
        conn.commit()


class ScriptedEmbeddings:
    """An embeddings client that returns the fixture's hand-placed query vectors.

    Only the query side is used by retrieval. An unmapped query is an error
    rather than a default vector: silently embedding to an arbitrary point would
    make a mistyped test query look like a retrieval failure.
    """

    def embed_query(self, text: str) -> list[float]:
        """Return the vector this fixture assigns to `text`."""
        try:
            return QUERY_VECTORS[text]
        except KeyError:
            raise KeyError(
                f"no scripted vector for {text!r} — add it to QUERY_VECTORS in "
                "tests/retrieval_corpus.py"
            ) from None

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Unused by retrieval; present so this satisfies the Embeddings protocol."""
        return [self.embed_query(text) for text in texts]
