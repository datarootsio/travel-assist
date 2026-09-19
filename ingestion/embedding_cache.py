"""Remember embeddings so a re-run costs nothing.

Embedding is metered per text, and providers cap the free tier hard — the free
tier used during development allowed 1,000 texts per day, which is about one
small ingestion. Early development needed three re-runs to shake out chunking bugs;
without a cache that would have been three days.

So every vector is stored on disk keyed by the text **and** the model **and** the
dimension. Re-ingesting an unchanged corpus becomes free, and only genuinely new
or edited chunks spend quota.

Including the model and dimension in the key is not defensive tidiness — it is
the whole safety property. A cache keyed on text alone would happily hand back
vectors from one model for a run configured with another: retrieval would still
return results, drawn from the wrong vector space, and nothing would report an
error. That is the same silent corruption the frozen-embedding guard exists to
prevent, and a cache is an easy place to reintroduce it.

The cache is a local build artefact, not course material: it is gitignored, and
deleting it only costs quota, never correctness.
"""

from __future__ import annotations

import hashlib
import json
import logging
import sqlite3
from collections.abc import Sequence
from pathlib import Path

logger = logging.getLogger(__name__)

DEFAULT_CACHE_PATH = Path("data/embedding_cache.sqlite")


class EmbeddingCache:
    """A content-addressed store of embedding vectors on disk."""

    def __init__(self, path: Path, *, model: str, dimension: int) -> None:
        """Open (or create) the cache for one specific model and width.

        Args:
            path: SQLite file to use.
            model: Embedding model name, part of every key.
            dimension: Vector width, also part of every key.
        """
        self.model = model
        self.dimension = dimension
        path.parent.mkdir(parents=True, exist_ok=True)
        self._connection = sqlite3.connect(path)
        self._connection.execute(
            "CREATE TABLE IF NOT EXISTS embeddings (key TEXT PRIMARY KEY, vector TEXT NOT NULL)"
        )
        self._connection.commit()

    def key_for(self, text: str) -> str:
        """Hash the text together with the model and width that will embed it."""
        material = f"{self.model}|{self.dimension}|{text}".encode()
        return hashlib.sha256(material).hexdigest()

    def get_many(self, texts: Sequence[str]) -> dict[str, list[float]]:
        """Return the cached vectors for whichever of `texts` are known.

        Returns:
            A mapping from text to vector, omitting anything not cached.
        """
        found: dict[str, list[float]] = {}
        if not texts:
            return found

        keys = {self.key_for(text): text for text in texts}
        placeholders = ",".join("?" * len(keys))
        rows = self._connection.execute(
            f"SELECT key, vector FROM embeddings WHERE key IN ({placeholders})",  # noqa: S608
            list(keys),
        ).fetchall()

        for key, vector in rows:
            found[keys[key]] = json.loads(vector)

        return found

    def put_many(self, vectors: dict[str, list[float]]) -> None:
        """Store vectors for their texts."""
        self._connection.executemany(
            "INSERT OR REPLACE INTO embeddings (key, vector) VALUES (?, ?)",
            [(self.key_for(text), json.dumps(vector)) for text, vector in vectors.items()],
        )
        self._connection.commit()

    def close(self) -> None:
        """Close the underlying database."""
        self._connection.close()

    def __enter__(self) -> EmbeddingCache:
        """Enter a context manager."""
        return self

    def __exit__(self, *exc_info: object) -> None:
        """Close on exit."""
        self.close()
