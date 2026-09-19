"""`pipeline/routing.py::classify_route` — the call, not the model.

Same shape as `test_steps.py`: the chat model is a system boundary, so a
`RunnableLambda` stands in for `get_chat_model().with_structured_output(...)`,
same `invoke(prompt) -> RouteDecision` contract, none of the network.
"""

from __future__ import annotations

from langchain_core.runnables import RunnableLambda

from travel_assist.pipeline.routing import RouteDecision, classify_route

DESTINATIONS = ["Porto", "Lisbon", "Marrakech"]


def scripted(decision: RouteDecision) -> RunnableLambda:
    """A structured model that returns `decision` regardless of what it is asked."""
    return RunnableLambda(lambda _prompt: decision)


def capturing(decision: RouteDecision) -> tuple[RunnableLambda, list[str]]:
    """A structured model that also records every prompt it was invoked with."""
    seen: list[str] = []

    def _invoke(prompt: str) -> RouteDecision:
        seen.append(prompt)
        return decision

    return RunnableLambda(_invoke), seen


def test_returns_the_models_decision_for_a_destination_question():
    decision = RouteDecision(route="destination_question", reason="Asks about Porto.")

    result = classify_route(
        "what's in Porto",
        structured_model=scripted(decision),
        destinations=DESTINATIONS,
    )

    assert result == decision


def test_returns_the_models_decision_for_out_of_scope():
    decision = RouteDecision(route="out_of_scope", reason="Barcelona is not in this corpus.")

    result = classify_route(
        "what neighbourhood should I stay in in Barcelona?",
        structured_model=scripted(decision),
        destinations=DESTINATIONS,
    )

    assert result == decision


def test_returns_the_models_decision_for_needs_clarification():
    decision = RouteDecision(route="needs_clarification", reason="No destination named.")

    result = classify_route(
        "where should I stay?",
        structured_model=scripted(decision),
        destinations=DESTINATIONS,
    )

    assert result == decision


def test_the_query_and_every_destination_reach_the_prompt():
    model, seen = capturing(RouteDecision(route="destination_question", reason="ok"))

    classify_route("what's in Porto", structured_model=model, destinations=DESTINATIONS)

    assert len(seen) == 1
    prompt = seen[0]
    assert "what's in Porto" in prompt
    for destination in DESTINATIONS:
        assert destination in prompt


def test_destinations_omitted_falls_back_to_the_indexs_own_contents(monkeypatch):
    """No `destinations=` means the router asks the index, not a hardcoded list.

    See `pipeline/routing.py`'s module docstring: a copy of the corpus's
    contents that could drift from what is actually indexed is exactly the
    silent-wrongness failure mode `db.check_embedding_contract` exists to
    catch elsewhere.
    """
    monkeypatch.setattr(
        "travel_assist.pipeline.routing.list_corpus_destinations",
        lambda **kwargs: ["Cuzco", "Yellowstone"],
    )
    model, seen = capturing(RouteDecision(route="destination_question", reason="ok"))

    classify_route("what's in Cuzco", structured_model=model)

    assert "Cuzco" in seen[0]
    assert "Yellowstone" in seen[0]
