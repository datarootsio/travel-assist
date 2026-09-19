"""`pipeline/context.py` — fitting retrieved chunks inside a token budget.

Pure function, no database, no model — a list of `RetrievedChunk` in, a shorter
list out. Expected values are hand-computed from the fixture's token counts, not
recomputed the way the code computes them.
"""

from __future__ import annotations

import logging

from travel_assist.pipeline.context import assemble_context
from travel_assist.retrieval.models import RetrievedChunk


def make_chunk(chunk_id: int, rank: int, token_count: int) -> RetrievedChunk:
    """A minimal chunk for budget tests — only `token_count` and rank matter."""
    return RetrievedChunk(
        chunk_id=chunk_id,
        title="Porto",
        url="https://en.wikivoyage.org/wiki/Porto",
        content=f"chunk {chunk_id}",
        token_count=token_count,
        score=1.0,
        rank=rank,
        method="hybrid",
    )


# Three chunks: 40 + 40 + 40 = 120 tokens if all kept.
THREE_CHUNKS = [make_chunk(1, 1, 40), make_chunk(2, 2, 40), make_chunk(3, 3, 40)]


def test_every_chunk_survives_when_the_budget_is_generous():
    assert assemble_context(THREE_CHUNKS, budget=1000) == THREE_CHUNKS


def test_nothing_survives_a_budget_smaller_than_the_first_chunk():
    assert assemble_context(THREE_CHUNKS, budget=10) == []


def test_included_is_a_prefix_of_the_ranking_not_a_scattered_subset():
    """A budget that fits chunks 1 and 3 but not 2 still stops at 2.

    Greedy, in rank order — not bin-packing to maximise budget utilisation.
    """
    chunks = [make_chunk(1, 1, 40), make_chunk(2, 2, 100), make_chunk(3, 3, 40)]

    assert assemble_context(chunks, budget=80) == [chunks[0]]


def test_a_budget_that_exactly_fits_keeps_everything():
    assert assemble_context(THREE_CHUNKS, budget=120) == THREE_CHUNKS


def test_an_empty_budget_keeps_nothing():
    assert assemble_context(THREE_CHUNKS, budget=0) == []


def test_no_chunks_in_means_no_chunks_out():
    assert assemble_context([], budget=1000) == []


def test_dropped_chunks_are_logged_with_what_and_why(caplog):
    with caplog.at_level(logging.INFO):
        assemble_context(THREE_CHUNKS, budget=50)  # only the first 40-token chunk fits

    assert len(caplog.records) == 1
    message = caplog.records[0].message
    assert "50" in message  # the budget
    assert "1/3" in message  # kept 1 of 3
    assert "Porto" in message  # names what was dropped


def test_nothing_is_logged_when_nothing_is_dropped(caplog):
    with caplog.at_level(logging.INFO):
        assemble_context(THREE_CHUNKS, budget=1000)

    assert caplog.records == []
