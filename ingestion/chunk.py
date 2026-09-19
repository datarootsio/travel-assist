"""Pack an article's sections into the passages that actually get retrieved.

INSTRUCTOR ONLY to run; worth reading closely, because chunking sets the ceiling
on how good retrieval can be. No amount of clever fusion recovers from passages
that split a thought in half.

Two decisions do the work, and only the second one happens in this file:

1. **Split on Wikivoyage's section structure first, then pack to a token
   budget.** `See`, `Eat`, `Sleep`, `Get around` are boundaries a traveller
   already thinks in, so chunks rarely straddle unrelated subjects. This part
   happens once, at corpus-build time, because `See > Museums`
   is a property of the *article*, not of how it later gets packed. By the time
   a `Section` reaches this file its markup is already stripped; this module
   only decides how much of it fits in one chunk.
2. **Prepend the section path before embedding.** Embedding
   the section path followed by the text, rather than the bare text, measurably
   improves retrieval: the vector then encodes the destination and the topic,
   not only the sentences. The path is *not* stored in `content`, because the
   answer should quote the prose, not the breadcrumb.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from functools import lru_cache

import tiktoken

# 400 tokens is close to one idea plus its immediate context — a museum with a
# sentence either side, not a bare fact. Larger chunks retrieve *fine*; they
# just spend the context budget on text the question never asked about, which
# is the exact cost the context-budget exercise asks participants to measure.
#
# 64-token overlap (16% of a chunk) is enough that a sentence spanning a
# boundary survives in one piece on at least one side, without paying for
# nearly a quarter of every chunk to be a repeat of its neighbour.
#
# Not free: changing either number invalidates every cached vector, because the
# embedding cache is keyed on the text actually sent to the provider.
DEFAULT_TARGET_TOKENS = 400
DEFAULT_OVERLAP_TOKENS = 64


@dataclass(frozen=True)
class Section:
    """A slice of an article under one heading path, already clean.

    Built during corpus-build, before this repo ships, and stored as
    part of `data/corpus.jsonl` — this module never parses wikitext.
    """

    path: str
    text: str


@dataclass(frozen=True)
class Chunk:
    """One retrievable passage.

    Attributes:
        chunk_index: Position within its article, used for stable ordering.
        section_path: Breadcrumb such as `"Barcelona > See > Museums"`.
        content: Clean prose. This is what an answer quotes and cites.
        token_count: Length of `content`, recorded for context budgeting.
    """

    chunk_index: int
    section_path: str
    content: str
    token_count: int

    @property
    def embedding_text(self) -> str:
        """What actually gets embedded: the path, then the prose.

        Deliberately different from `content`. See the module docstring.
        """
        return f"{self.section_path}\n\n{self.content}"


@lru_cache(maxsize=1)
def _encoder() -> tiktoken.Encoding:
    # cl100k_base is close enough for budgeting across the models in play here.
    return tiktoken.get_encoding("cl100k_base")


def count_tokens(text: str) -> int:
    """Count tokens the way the context budget will."""
    return len(_encoder().encode(text))


def _pack(text: str, *, target_tokens: int, overlap_tokens: int) -> list[str]:
    """Split `text` into overlapping windows of at most `target_tokens`.

    Overlap exists so a sentence that answers the question cannot be halved by a
    chunk boundary — the classic way a corpus becomes unable to answer something
    it plainly contains.
    """
    encoder = _encoder()
    tokens = encoder.encode(text)

    if len(tokens) <= target_tokens:
        return [text]

    stride = max(target_tokens - overlap_tokens, 1)
    windows = []
    for start in range(0, len(tokens), stride):
        window = tokens[start : start + target_tokens]
        if not window:
            break
        windows.append(encoder.decode(window).strip())
        if start + target_tokens >= len(tokens):
            break

    return [window for window in windows if window]


def chunk_article(
    sections: Sequence[Section],
    *,
    target_tokens: int = DEFAULT_TARGET_TOKENS,
    overlap_tokens: int = DEFAULT_OVERLAP_TOKENS,
) -> list[Chunk]:
    """Pack an article's already-clean sections into retrievable chunks.

    Args:
        sections: The article's sections, in document order — as stored in
            `data/corpus.jsonl`, already split and cleaned before this repo
            ships.
        target_tokens: Upper bound on chunk size.
        overlap_tokens: How much consecutive chunks share.

    Returns:
        Chunks in document order with sequential `chunk_index` across the whole
        article, not restarting per section. An article with no sections yields
        an empty list rather than an empty chunk.
    """
    chunks: list[Chunk] = []

    for section in sections:
        for piece in _pack(
            section.text, target_tokens=target_tokens, overlap_tokens=overlap_tokens
        ):
            chunks.append(
                Chunk(
                    chunk_index=len(chunks),
                    section_path=section.path,
                    content=piece,
                    token_count=count_tokens(piece),
                )
            )

    return chunks
