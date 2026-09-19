"""Turn a ranked list of retrieved chunks into what actually reaches the prompt.

Diff from hybrid-search: retrieval's output is no longer the end of the story —
it has to fit inside `CONTEXT_TOKEN_BUDGET`, and what gets left out has to be
visible, not just silently gone.

Retrieval already over-fetches on purpose (`hybrid_search`'s `vector_k`/`keyword_k`
default to 30 each), so assembly almost always has more candidates than the budget
can hold. Two decisions follow from that:

**Greedy, in rank order, stop at the first miss.** Chunks arrive best-first from
`hybrid_search`; taking them in that order until one does not fit, then stopping,
is simpler to reason about than a bin-packing scheme that skips a chunk that
doesn't fit to keep trying smaller ones further down the ranking — and it keeps
"included" a strict prefix of "ranked", which is what makes the drop log's *"you
retrieved eight, used two — what did the other six cost you?"* framing legible.

**Token counts are already known.** `RetrievedChunk.token_count` was computed
once, at ingestion time, over the exact `content` an answer will quote — see
`ingestion/chunk.py::count_tokens`. Assembly sums that field; it never
re-tokenizes anything.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence

from travel_assist.retrieval.models import RetrievedChunk

logger = logging.getLogger(__name__)


def assemble_context(chunks: Sequence[RetrievedChunk], *, budget: int) -> list[RetrievedChunk]:
    """Keep as many top-ranked chunks as fit within `budget` tokens.

    Args:
        chunks: Best-first, e.g. `hybrid_search`'s output. Ranking, not content,
            decides what survives.
        budget: Maximum combined `token_count` across the returned chunks.

    Returns:
        A prefix of `chunks`: every chunk up to, but excluding, the first one
        that would push the running total over `budget`. Logs one line naming
        what was dropped and why, whenever the prefix is shorter than `chunks`.
    """
    raise NotImplementedError("context-budget — see the docstring above")
