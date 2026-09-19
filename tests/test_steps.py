"""`pipeline/steps.py::generate_answer` — the call, not the model.

The chat model is a system boundary, so it is injected rather than faked by
mocking `travel_assist.models`. A `RunnableLambda` stands in for
`get_chat_model().with_structured_output(...)`: same `invoke(prompt) -> DestinationAnswer`
contract, none of the network.
"""

from __future__ import annotations

import pytest
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables import RunnableLambda

from travel_assist.pipeline.citations import Claim, DestinationAnswer, UngroundedCitationError
from travel_assist.pipeline.context import assemble_context
from travel_assist.pipeline.steps import generate_answer
from travel_assist.retrieval.models import RetrievedChunk
from travel_assist.retrieval.search import hybrid_search

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


def scripted(answer: DestinationAnswer) -> RunnableLambda:
    """A structured model that returns `answer` regardless of what it is asked."""
    return RunnableLambda(lambda _prompt: answer)


def capturing(answer: DestinationAnswer) -> tuple[RunnableLambda, list[str]]:
    """A structured model that also records every prompt it was invoked with."""
    seen: list[str] = []

    def _invoke(prompt: str) -> DestinationAnswer:
        seen.append(prompt)
        return answer

    return RunnableLambda(_invoke), seen


def capturing_any(answer: DestinationAnswer) -> tuple[RunnableLambda, list[object]]:
    """Like `capturing`, but records whatever it was invoked with, string or message list."""
    seen: list[object] = []

    def _invoke(model_input: object) -> DestinationAnswer:
        seen.append(model_input)
        return answer

    return RunnableLambda(_invoke), seen


def test_returns_the_models_answer_when_every_citation_is_grounded():
    answer = DestinationAnswer(claims=[Claim(text="It has a bookshop.", source_chunk_id=1)])

    result = generate_answer("what's in Porto", CHUNKS, structured_model=scripted(answer))

    assert result == answer


def test_a_hallucinated_citation_is_caught_not_silently_returned():
    answer = DestinationAnswer(claims=[Claim(text="Made up.", source_chunk_id=999)])

    with pytest.raises(UngroundedCitationError):
        generate_answer("what's in Porto", CHUNKS, structured_model=scripted(answer))


def test_the_query_and_every_chunks_content_reach_the_prompt():
    model, seen = capturing(DestinationAnswer(claims=[]))

    generate_answer("what's in Porto", CHUNKS, structured_model=model)

    assert len(seen) == 1
    prompt = seen[0]
    assert "what's in Porto" in prompt
    assert "Livraria Lello" in prompt
    assert "francesinha" in prompt
    assert "chunk_id=1" in prompt
    assert "chunk_id=2" in prompt


def test_a_chunk_left_out_of_context_never_reaches_the_prompt():
    """Only what's passed as `context` is visible.

    This is what makes the citation check meaningful: the model literally
    cannot see chunk 2.
    """
    model, seen = capturing(DestinationAnswer(claims=[]))

    generate_answer("what's in Porto", [CHUNKS[0]], structured_model=model)

    assert "francesinha" not in seen[0]


def test_a_custom_prompt_template_is_used_instead_of_the_default():
    model, seen = capturing(DestinationAnswer(claims=[]))

    generate_answer(
        "what's in Porto",
        CHUNKS,
        structured_model=model,
        prompt_template="CUSTOM: {query} // {context}",
    )

    assert seen[0].startswith("CUSTOM: what's in Porto //")


# --------------------------------------------------------------------------
# history — prior turns as real messages, not baked into query or context
# --------------------------------------------------------------------------


def test_no_history_invokes_with_a_bare_string_exactly_as_before():
    """Empty history sends a bare string, not a message list."""
    model, seen = capturing_any(DestinationAnswer(claims=[]))

    generate_answer("what's in Porto", CHUNKS, structured_model=model)

    assert isinstance(seen[0], str)


def test_history_is_sent_as_real_messages_ahead_of_the_formatted_prompt():
    model, seen = capturing_any(DestinationAnswer(claims=[]))
    history = [HumanMessage("Tell me about Porto"), AIMessage("Porto has a bookshop.")]

    generate_answer("what about the food?", CHUNKS, history=history, structured_model=model)

    assert len(seen) == 1
    invoked = seen[0]
    assert isinstance(invoked, list)
    assert invoked[:2] == history
    assert isinstance(invoked[2], HumanMessage)
    assert "what about the food?" in invoked[2].content
    assert "francesinha" in invoked[2].content


# --------------------------------------------------------------------------
# Live: the scope-check paragraph, against the real model and real retrieval
# --------------------------------------------------------------------------
#
# Barcelona is outside the corpus; hybrid_search still returns its nearest
# (wrong-destination) chunks. This is a prompt-level mitigation, not airtight,
# so only the reliable part is asserted: at least one
# claim names the mismatch. The reliable fix is `routing`'s scope check before
# retrieval ever runs.


@pytest.mark.live
def test_a_question_about_a_destination_outside_the_corpus_gets_a_disclaimer():
    query = "what neighbourhood should I stay in in Barcelona?"
    chunks = hybrid_search(query, k=5)
    context = assemble_context(chunks, budget=2000)

    answer = generate_answer(query, context)

    assert any(
        "barcelona" in claim.text.lower()
        and any(negation in claim.text.lower() for negation in ("not", "n't", "instead"))
        for claim in answer.claims
    ), f"no claim flagged the destination mismatch: {[c.text for c in answer.claims]}"
