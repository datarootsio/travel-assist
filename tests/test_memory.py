"""`pipeline/memory.py` — a conversation that shrinks itself when it grows too large.

`summarise` is the only thing here that reaches a provider, and it is injected
with `FakeListChatModel` — a scripted client with a known response, not a
pretend backend. Everything else is a pure function, or `CompactingHistory`'s
own logic layered on LangChain's already-tested `InMemoryChatMessageHistory`,
so none of it needs a database or a network.
"""

from __future__ import annotations

import pytest
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from travel_assist.pipeline.memory import (
    CompactingHistory,
    count_tokens,
    drop_oldest,
    history_token_count,
    summarise,
)


def turn(text: str) -> list[HumanMessage | AIMessage]:
    """One human/assistant exchange, so a test conversation reads as N `turn()` calls."""
    return [HumanMessage(text), AIMessage(f"Reply to: {text}")]


# --------------------------------------------------------------------------
# history_token_count — the shared measurement both strategies budget against
# --------------------------------------------------------------------------


def test_history_token_count_sums_every_message():
    messages = [HumanMessage("hello"), AIMessage("hi there")]
    expected = count_tokens("hello") + count_tokens("hi there")

    assert history_token_count(messages) == expected


def test_history_token_count_of_no_messages_is_zero():
    assert history_token_count([]) == 0


# --------------------------------------------------------------------------
# drop_oldest — pure, no model call
# --------------------------------------------------------------------------


def test_drop_oldest_is_a_noop_when_already_within_budget():
    messages = [HumanMessage("hi"), AIMessage("hello")]
    budget = history_token_count(messages) + 100

    assert drop_oldest(messages, budget=budget) == messages


def test_drop_oldest_drops_the_oldest_messages_first():
    """Ten turns, a budget that only fits the last two — recency survives."""
    messages: list[HumanMessage | AIMessage] = [m for i in range(10) for m in turn(f"q{i}")]
    budget = history_token_count(messages[-2:])

    result = drop_oldest(messages, budget=budget)

    assert result == messages[-2:]


def test_drop_oldest_never_drops_a_leading_system_message():
    """A budget of zero would drop everything else, but never the system prompt."""
    messages: list[SystemMessage | HumanMessage | AIMessage] = [
        SystemMessage("You are a travel assistant."),
        *turn("where is Porto"),
        *turn("what about Lisbon"),
    ]

    result = drop_oldest(messages, budget=0)

    assert result == [messages[0]]


def test_drop_oldest_result_fits_the_budget_when_the_corpus_allows_it():
    messages: list[HumanMessage | AIMessage] = [m for i in range(20) for m in turn(f"q{i}")]
    budget = 50

    result = drop_oldest(messages, budget=budget)

    assert history_token_count(result) <= budget
    assert result  # the budget is not so tight that even one message can't fit


# --------------------------------------------------------------------------
# summarise — the one function that reaches a (scripted) model
# --------------------------------------------------------------------------


def test_summarise_keeps_the_most_recent_messages_verbatim():
    messages: list[HumanMessage | AIMessage] = [m for i in range(10) for m in turn(f"q{i}")]
    fake_model = FakeListChatModel(responses=["everything before this was about destinations"])

    result = summarise(messages, keep_recent=4, chat_model=fake_model)

    assert result[-4:] == messages[-4:]


def test_summarise_replaces_older_messages_with_exactly_one_summary_message():
    messages: list[HumanMessage | AIMessage] = [m for i in range(10) for m in turn(f"q{i}")]
    fake_model = FakeListChatModel(responses=["a summary of the earlier turns"])

    result = summarise(messages, keep_recent=4, chat_model=fake_model)

    assert len(result) == 1 + 4  # one summary message, plus the kept recent ones
    assert isinstance(result[0], SystemMessage)
    assert "a summary of the earlier turns" in result[0].content


def test_summarise_is_a_noop_and_calls_the_model_zero_times_when_nothing_is_old_enough():
    messages: list[HumanMessage | AIMessage] = [m for i in range(2) for m in turn(f"q{i}")]
    fake_model = FakeListChatModel(responses=["should never be used"])

    result = summarise(messages, keep_recent=10, chat_model=fake_model)

    assert result == messages
    assert fake_model.i == 0  # FakeListChatModel's own call counter


# --------------------------------------------------------------------------
# CompactingHistory — storage inherited from LangChain, compaction is ours
# --------------------------------------------------------------------------


def test_add_messages_is_a_noop_beyond_storing_when_under_budget():
    messages = [HumanMessage("hi"), AIMessage("hello")]
    budget = history_token_count(messages) + 100

    for strategy in ("summarise", "drop_oldest", "none"):
        history = CompactingHistory(budget=budget, strategy=strategy)
        history.add_messages(messages)
        assert history.messages == messages


def test_strategy_none_grows_unboundedly():
    """The "degrades without compaction" half of the compaction verify step."""
    history = CompactingHistory(budget=200, strategy="none")
    token_counts = []

    for i in range(30):
        history.add_messages(turn(f"question number {i} about a different destination entirely"))
        token_counts.append(history_token_count(history.messages))

    assert token_counts[-1] > 200
    assert token_counts == sorted(token_counts)  # strictly non-decreasing — nothing is ever shed


@pytest.mark.parametrize("strategy", ["summarise", "drop_oldest"])
def test_compaction_recovers_and_stays_bounded(strategy: str):
    """The "recovers with it" half — flip the selector, same property holds either way."""
    fake_model = FakeListChatModel(responses=[f"summary #{i}" for i in range(30)])
    history = CompactingHistory(budget=200, strategy=strategy, chat_model=fake_model)

    for i in range(30):
        history.add_messages(turn(f"question number {i} about a different destination entirely"))
        assert history_token_count(history.messages) <= 200
