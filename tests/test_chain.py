"""`pipeline/chain.py::answer_query` — the whole day-1 pipeline, one call.

The property that matters most: `out_of_scope` and `needs_clarification` must
never call `hybrid_search`. Proved the hard way here, not just counted — the
patched `hybrid_search` raises if it is ever reached on those branches, which
is a stronger guarantee than a call count of zero would be.

Everything is injected (`route_model`, `structured_model`, `destinations`), so
none of this needs a database, an embedding client or a live chat model.
"""

from __future__ import annotations

from typing import Any
from uuid import UUID

import pytest
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables import RunnableLambda

from travel_assist.pipeline.chain import (
    PipelineTurn,
    _display_names,
    answer_query,
    condense_question,
)
from travel_assist.pipeline.citations import Claim, DestinationAnswer, self_check
from travel_assist.pipeline.memory import CompactingHistory
from travel_assist.pipeline.routing import RouteDecision
from travel_assist.retrieval.models import RetrievedChunk

DESTINATIONS = ["Lisbon", "Porto"]

CHUNKS = [
    RetrievedChunk(
        chunk_id=1,
        title="Porto",
        url="https://en.wikivoyage.org/wiki/Porto",
        section_path="Porto > See",
        content="Livraria Lello is a neo-gothic bookshop.",
        token_count=10,
        score=1.0,
        rank=1,
        method="hybrid",
    ),
    RetrievedChunk(
        chunk_id=2,
        title="Porto",
        url="https://en.wikivoyage.org/wiki/Porto",
        section_path="Porto > Eat",
        content="The francesinha is a sandwich under melted cheese.",
        token_count=10,
        score=0.9,
        rank=2,
        method="hybrid",
    ),
]


def scripted_route(decision: RouteDecision) -> RunnableLambda:
    """A routing model that returns `decision` regardless of what it is asked."""
    return RunnableLambda(lambda _prompt: decision)


def capturing_route(decision: RouteDecision) -> tuple[RunnableLambda, list[str]]:
    """A routing model that also records every prompt it was invoked with."""
    seen: list[str] = []

    def _invoke(prompt: str) -> RouteDecision:
        seen.append(prompt)
        return decision

    return RunnableLambda(_invoke), seen


def scripted_answer(answer: DestinationAnswer) -> RunnableLambda:
    """A structured model that returns `answer` regardless of what it is asked."""
    return RunnableLambda(lambda _prompt: answer)


def refusing_hybrid_search(*_args: object, **_kwargs: object) -> list[RetrievedChunk]:
    """Stands in for `hybrid_search` on branches that must never call it."""
    raise AssertionError("hybrid_search must not be called")


def counting_hybrid_search(monkeypatch: object, chunks: list[RetrievedChunk] = CHUNKS) -> list[str]:
    """Patch `chain.hybrid_search` with a spy; return the list it records calls into."""
    calls: list[str] = []

    def spy(query: str, **_kwargs: object) -> list[RetrievedChunk]:
        calls.append(query)
        return chunks

    monkeypatch.setattr("travel_assist.pipeline.chain.hybrid_search", spy)  # type: ignore[attr-defined]
    return calls


# --------------------------------------------------------------------------
# _display_names — district articles collapse to their city for display
# --------------------------------------------------------------------------


def test_display_names_collapses_district_articles_to_their_city():
    raw = ["Cape Town", "Lisbon", "Lisbon/Alfama", "Lisbon/Belém", "Porto"]

    assert _display_names(raw) == ["Cape Town", "Lisbon", "Porto"]


def test_the_refusal_message_shows_cities_not_district_articles():
    destinations_with_districts = ["Lisbon", "Lisbon/Alfama", "Lisbon/Baixa", "Porto"]
    decision = RouteDecision(route="out_of_scope", reason="Barcelona is not in this corpus.")

    turn = answer_query(
        "what neighbourhood should I stay in in Barcelona?",
        route_model=scripted_route(decision),
        destinations=destinations_with_districts,
    )

    assert turn.message is not None
    assert "Lisbon/Alfama" not in turn.message
    assert "Lisbon/Baixa" not in turn.message
    assert turn.message.count("Lisbon") == 1


# --------------------------------------------------------------------------
# out_of_scope — refuses without ever calling hybrid_search
# --------------------------------------------------------------------------


def test_out_of_scope_never_calls_hybrid_search_and_refuses(monkeypatch):
    monkeypatch.setattr("travel_assist.pipeline.chain.hybrid_search", refusing_hybrid_search)
    decision = RouteDecision(route="out_of_scope", reason="Barcelona is not in this corpus.")

    turn = answer_query(
        "what neighbourhood should I stay in in Barcelona?",
        route_model=scripted_route(decision),
        destinations=DESTINATIONS,
    )

    assert turn.route == "out_of_scope"
    assert turn.retrieved is False
    assert turn.answer is None
    assert turn.message is not None
    assert "Barcelona is not in this corpus." in turn.message
    assert "Lisbon" in turn.message
    assert "Porto" in turn.message


# --------------------------------------------------------------------------
# needs_clarification — also never calls hybrid_search
# --------------------------------------------------------------------------


def test_needs_clarification_never_calls_hybrid_search_and_asks(monkeypatch):
    monkeypatch.setattr("travel_assist.pipeline.chain.hybrid_search", refusing_hybrid_search)
    decision = RouteDecision(route="needs_clarification", reason="no destination named")

    turn = answer_query(
        "where should I stay?",
        route_model=scripted_route(decision),
        destinations=DESTINATIONS,
    )

    assert turn.route == "needs_clarification"
    assert turn.retrieved is False
    assert turn.answer is None
    assert turn.message is not None


# --------------------------------------------------------------------------
# destination_question — retrieves exactly once, self-check attached
# --------------------------------------------------------------------------


def test_destination_question_retrieves_exactly_once_and_self_checks(monkeypatch):
    calls = counting_hybrid_search(monkeypatch)
    decision = RouteDecision(route="destination_question", reason="about Porto")
    answer = DestinationAnswer(claims=[Claim(text="It has a bookshop.", source_chunk_id=1)])

    turn = answer_query(
        "what's in Porto",
        route_model=scripted_route(decision),
        structured_model=scripted_answer(answer),
        destinations=DESTINATIONS,
    )

    assert len(calls) == 1
    assert turn.route == "destination_question"
    assert turn.retrieved is True
    assert turn.answer == answer
    assert turn.context == CHUNKS
    assert turn.self_check == self_check(answer)


def test_self_check_flags_an_uncited_claim(monkeypatch):
    counting_hybrid_search(monkeypatch)
    decision = RouteDecision(route="destination_question", reason="about Porto")
    uncited_claim = Claim(text="Some unsupported claim.", source_chunk_id=None)
    answer = DestinationAnswer(claims=[uncited_claim])

    turn = answer_query(
        "what's in Porto",
        route_model=scripted_route(decision),
        structured_model=scripted_answer(answer),
        destinations=DESTINATIONS,
    )

    assert turn.self_check is not None
    assert turn.self_check.passed is False
    assert uncited_claim in turn.self_check.uncited


# --------------------------------------------------------------------------
# PipelineTurn.display_text
# --------------------------------------------------------------------------


def test_display_text_is_the_answers_claims_joined():
    answer = DestinationAnswer(
        claims=[
            Claim(text="Claim one.", source_chunk_id=1),
            Claim(text="Claim two.", source_chunk_id=2),
        ]
    )
    turn = PipelineTurn(
        route="destination_question", retrieved=True, answer=answer, self_check=self_check(answer)
    )

    assert turn.display_text == "Claim one. Claim two."


def test_display_text_is_the_message_for_refusal_and_clarification():
    refusal = PipelineTurn(route="out_of_scope", message="no.")
    clarification = PipelineTurn(route="needs_clarification", message="which one?")

    assert refusal.display_text == "no."
    assert clarification.display_text == "which one?"


# --------------------------------------------------------------------------
# History: stores the raw query, reads recent turns back for coreference
# --------------------------------------------------------------------------


def test_history_gets_the_raw_query_and_the_turns_display_text(monkeypatch):
    counting_hybrid_search(monkeypatch)
    decision = RouteDecision(route="destination_question", reason="about Porto")
    answer = DestinationAnswer(claims=[Claim(text="It has a bookshop.", source_chunk_id=1)])
    history = CompactingHistory(budget=10_000, strategy="none")

    turn = answer_query(
        "what's in Porto",
        history=history,
        route_model=scripted_route(decision),
        structured_model=scripted_answer(answer),
        destinations=DESTINATIONS,
    )

    assert len(history.messages) == 2
    assert history.messages[0] == HumanMessage("what's in Porto")
    assert history.messages[1] == AIMessage(turn.display_text)


def test_a_condensed_standalone_question_reaches_retrieval_and_routing(monkeypatch):
    """Coreference resolution happens once, via `condense_question`, not by text-gluing.

    A follow-up like "what about the food there?" needs the prior turn to
    resolve "there" — proved by inspecting what `hybrid_search` and
    `classify_route` actually received: the condensed rewrite, not the raw
    query with history stitched onto it.
    """
    calls = counting_hybrid_search(monkeypatch)
    condensed = "What food is there in Porto?"
    route_model, route_prompts_seen = capturing_route(
        RouteDecision(route="destination_question", reason="about Porto")
    )
    answer = DestinationAnswer(claims=[Claim(text="Try the francesinha.", source_chunk_id=2)])
    history = CompactingHistory(budget=10_000, strategy="none")
    history.add_messages([HumanMessage("Tell me about Porto"), AIMessage("Porto has a bookshop.")])

    answer_query(
        "what about the food there?",
        history=history,
        route_model=route_model,
        structured_model=scripted_answer(answer),
        condense_chat_model=FakeListChatModel(responses=[condensed]),
        destinations=DESTINATIONS,
    )

    assert calls == [condensed]
    assert len(route_prompts_seen) == 1
    assert condensed in route_prompts_seen[0]


# --------------------------------------------------------------------------
# condense_question — the retrieval-side rewrite, a real prompt|model|parser chain
# --------------------------------------------------------------------------


def test_condense_question_is_a_noop_and_calls_no_model_when_history_is_empty():
    result = condense_question("what's in Porto", [])

    assert result == "what's in Porto"


def test_condense_question_invokes_the_model_and_returns_its_rewrite_stripped():
    fake_model = FakeListChatModel(responses=["  What food is there in Porto?  "])
    history = [HumanMessage("Tell me about Porto"), AIMessage("Porto has a bookshop.")]

    result = condense_question("what about the food there?", history, chat_model=fake_model)

    assert result == "What food is there in Porto?"


# --------------------------------------------------------------------------
# Live: an out-of-corpus destination is refused before retrieval runs
# (the reliable fix, replacing an earlier prompt-level mitigation)
# --------------------------------------------------------------------------


@pytest.mark.live
def test_a_destination_outside_the_corpus_is_refused_without_retrieving_live():
    turn = answer_query("what neighbourhood should I stay in in Barcelona?")

    assert turn.route == "out_of_scope"
    assert turn.retrieved is False
    assert turn.answer is None


# --------------------------------------------------------------------------
# One runnable — routing runs inside the traced pipeline, not before it
# --------------------------------------------------------------------------


class _RunRecorder(BaseCallbackHandler):
    """Records every chain run's name and whether it had a parent run."""

    def __init__(self) -> None:
        self.runs: list[tuple[str | None, bool]] = []

    def on_chain_start(
        self,
        serialized: dict[str, Any],
        inputs: dict[str, Any],
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> None:
        self.runs.append((kwargs.get("name"), parent_run_id is not None))


def test_routing_runs_inside_the_one_traced_pipeline(monkeypatch):
    counting_hybrid_search(monkeypatch)
    decision = RouteDecision(route="destination_question", reason="about Porto")
    answer = DestinationAnswer(claims=[Claim(text="It has a bookshop.", source_chunk_id=1)])
    recorder = _RunRecorder()

    answer_query(
        "what's in Porto",
        route_model=scripted_route(decision).with_config(run_name="route_model"),
        structured_model=scripted_answer(answer),
        destinations=DESTINATIONS,
        config={"callbacks": [recorder]},
    )

    assert ("route_model", True) in recorder.runs  # a child of the pipeline run
    assert sum(1 for _name, has_parent in recorder.runs if not has_parent) == 1  # one root
