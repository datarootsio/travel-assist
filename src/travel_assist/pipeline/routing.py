"""Diff from history/compaction: before anything is retrieved, the question is classified.

`pipeline/chain.py` wires this into a `RunnableBranch`, and only one of its
three branches ever calls `hybrid_search`.

Scope is checked against the index's actual contents
(`db.list_corpus_destinations`), not a list of destination names hardcoded
into the prompt. The corpus this course ships with is fixed, but a copy of its
contents that could drift from what is actually indexed is exactly the kind of
silent wrongness `db.check_embedding_contract` exists to catch elsewhere — the
router should not reintroduce it one file over.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal, cast

from langchain_core.language_models import LanguageModelInput
from langchain_core.runnables import Runnable
from pydantic import BaseModel

from travel_assist.config import Settings, get_settings
from travel_assist.db import list_corpus_destinations
from travel_assist.models import get_chat_model

Route = Literal["destination_question", "out_of_scope", "needs_clarification"]

_ROUTING_PROMPT = """You route questions for a travel assistant, before any \
retrieval happens. Get this right: retrieval always returns its closest \
matches even when nothing relevant exists, so a wrong route here becomes a \
wrong answer later that looks confident and cited.

The corpus covers exactly these destinations, and nothing else:
{destinations}

Classify the question as exactly one of:
- "destination_question": about one or more of the destinations listed above.
- "out_of_scope": names a destination that is NOT in the list above, or asks \
for something the corpus cannot answer regardless of destination — live \
prices, availability, visas, or booking a trip.
- "needs_clarification": no destination is named or inferable from the \
question, so there is nothing to search for yet.

Question: {query}

Give the route and one sentence explaining it."""


class RouteDecision(BaseModel):
    """Where a question should go, and why.

    Attributes:
        route: `"destination_question"` if `query` is answerable from the
            corpus; `"out_of_scope"` if it names a destination the corpus does
            not cover, or asks for something retrieval cannot answer regardless
            of destination; `"needs_clarification"` if no destination is named
            or inferable.
        reason: One sentence. Shown to the user for `out_of_scope` and
            `needs_clarification` — a refusal or a clarifying question with no
            reason attached is not a useful answer either.
    """

    route: Route
    reason: str


def classify_route(
    query: str,
    *,
    structured_model: Runnable[LanguageModelInput, RouteDecision] | None = None,
    settings: Settings | None = None,
    destinations: Sequence[str] | None = None,
) -> RouteDecision:
    """Decide whether `query` should be answered, refused, or clarified.

    Args:
        query: The user's question, forwarded into the prompt unchanged.
        structured_model: A `Runnable` returning `RouteDecision`, used instead
            of building one from `settings`. Injected by tests with a scripted
            double; built from `get_chat_model()` otherwise.
        settings: Configuration; defaults to the process-wide Settings.
        destinations: The corpus's known destinations, listed in the prompt.
            Injected by tests; queried fresh via `list_corpus_destinations`
            otherwise.

    Returns:
        The model's routing decision.
    """
    raise NotImplementedError("routing — see the docstring above")
