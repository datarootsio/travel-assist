"""`hybrid_search` is travel-assist's frozen retrieval entrypoint.

Its signature and location are frozen once it lands: the day-1 pipeline calls it
directly, and it is the function any downstream consumer — including day 2's
agent work, which continues in its own repo cloned from this one — imports and
calls unmodified. This test is deliberately implementation-coupled, in the same
spirit as `test_no_langgraph.py` — pinning a signature is its entire job, and it
must break the moment someone changes it, before nineteen people find out on
the day.
"""

from __future__ import annotations

import inspect
from typing import get_type_hints

from travel_assist.retrieval.models import RetrievedChunk
from travel_assist.retrieval.search import hybrid_search

# Everything a caller may depend on: a decorator and an args schema around this
# prefix, nothing else. Anything after `filters` is retrieval plumbing (settings,
# embeddings) a wrapper never passes, so it is not pinned.
FROZEN_PARAMETERS = ["query", "k", "vector_k", "keyword_k", "filters"]


def test_hybrid_search_lives_where_callers_expect_to_import_it():
    assert hybrid_search.__module__ == "travel_assist.retrieval.search"


def test_the_frozen_parameters_keep_their_names_and_order():
    names = [p.name for p in inspect.signature(hybrid_search).parameters.values()]
    assert names[: len(FROZEN_PARAMETERS)] == FROZEN_PARAMETERS


def test_query_and_k_are_positional_or_keyword_the_rest_are_keyword_only():
    parameters = inspect.signature(hybrid_search).parameters

    assert parameters["query"].kind == inspect.Parameter.POSITIONAL_OR_KEYWORD
    assert parameters["k"].kind == inspect.Parameter.POSITIONAL_OR_KEYWORD
    for name in ("vector_k", "keyword_k", "filters"):
        assert parameters[name].kind == inspect.Parameter.KEYWORD_ONLY, (
            f"{name} must be keyword-only — the `*` in the frozen signature"
        )


def test_the_frozen_defaults():
    parameters = inspect.signature(hybrid_search).parameters

    assert parameters["query"].default is inspect.Parameter.empty, "query must be required"
    assert parameters["k"].default == 8
    assert parameters["vector_k"].default == 30
    assert parameters["keyword_k"].default == 30
    assert parameters["filters"].default is None


def test_any_parameter_after_filters_is_optional_plumbing():
    """Extra injection points may exist, but a caller must never be forced to pass one.

    Everything past the frozen prefix (settings, embeddings, ...) has to default
    to something that works with zero arguments.
    """
    parameters = list(inspect.signature(hybrid_search).parameters.values())
    for parameter in parameters[len(FROZEN_PARAMETERS) :]:
        assert parameter.kind == inspect.Parameter.KEYWORD_ONLY
        assert parameter.default is None


def test_hybrid_search_returns_a_list_of_retrieved_chunks():
    hints = get_type_hints(hybrid_search)
    assert hints["return"] == list[RetrievedChunk]
