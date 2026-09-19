"""Conversation history that persists across turns, and shrinks itself when it grows too large.

Diff from context-budget/structured-answer: a participant can now ask a
follow-up without the transcript blowing the token budget the model call
itself needs.

Four decisions worth the minute they take:

**Storage is not hand-written, threading is.** `CompactingHistory` extends
LangChain's own `InMemoryChatMessageHistory` rather than reimplementing a list
with append/read methods — that plumbing already exists and hiding it would
teach nothing. What *is* worth writing by hand is the one thing LangChain
doesn't do for you: deciding when a transcript is too big and picking a
strategy to shrink it. `CompactingHistory` overrides exactly one method,
`add_message` (singular), to hook that decision into the one place new
messages arrive.

**Not wired through `RunnableWithMessageHistory`, deliberately.** That class
exists for exactly this — reading history for a session, injecting it, and
appending the new turn — but it is deprecated as of `langchain-core` 1.3.3,
scheduled for removal in 2.0.0, and its own deprecation message names its
replacement: LangGraph's persistence. Adopting the replacement would import
LangGraph into day-1 code, which this repo's LangChain-only rule forbids outright.
So `CompactingHistory` is called directly — `history.add_messages([...])`,
then read `history.messages` — by whatever composes the day-1 chain.
Checked against the installed package, not recalled from memory or a
tutorial; see the constructor docstring below for the deprecation warning's
exact text.

**`drop_oldest` never discards a leading `SystemMessage`.** That message is
the assistant's own instructions, not part of the conversation being
trimmed — losing it mid-conversation would silently change the assistant's
behaviour, which is a worse failure than a slightly larger transcript.

**Token counting here is a second, near-identical copy of
`ingestion/chunk.py::count_tokens`,** not a shared import. `ingestion` is
instructor-only tooling that runs once at corpus-build time; this module is
served-app code a participant's `travel-assist chat` process imports on every
turn. Three duplicated lines is cheaper than a dependency between those two
layers.
"""

from __future__ import annotations

from collections.abc import Sequence
from functools import lru_cache
from typing import Any, cast

import tiktoken
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from pydantic import PrivateAttr

from travel_assist.config import Settings, get_settings
from travel_assist.models import get_chat_model

# How many of the most recent messages `summarise` keeps verbatim. Small
# enough that a summarisation call is worth its cost; large enough that the
# immediately preceding turn — the one a follow-up question usually refers
# to — is never itself the thing being summarised away.
DEFAULT_KEEP_RECENT = 4

_SUMMARY_INSTRUCTION = (
    "Summarise the conversation so far in a few sentences. Keep any destination "
    "names, dates, preferences or constraints the traveller mentioned — the "
    "summary replaces the messages it covers, so anything it drops is gone."
)


@lru_cache(maxsize=1)
def _encoder() -> tiktoken.Encoding:
    # cl100k_base is close enough for budgeting across the models in play here —
    # same choice, and same reasoning, as ingestion/chunk.py.
    return tiktoken.get_encoding("cl100k_base")


def count_tokens(text: str) -> int:
    """Count tokens the way the history budget will."""
    return len(_encoder().encode(text))


def history_token_count(messages: Sequence[BaseMessage]) -> int:
    """Total tokens across every message's content.

    Course messages are always plain text — no multimodal content — so a
    message's `content` is read as `str` directly rather than handling the
    richer content-block shape `BaseMessage.content` allows in general.
    """
    return sum(count_tokens(cast(str, message.content)) for message in messages)


def summarise(
    messages: Sequence[BaseMessage],
    *,
    keep_recent: int = DEFAULT_KEEP_RECENT,
    chat_model: BaseChatModel | None = None,
    settings: Settings | None = None,
) -> list[BaseMessage]:
    """Replace everything except the most recent `keep_recent` messages with one summary.

    Args:
        messages: The full conversation so far, oldest first.
        keep_recent: How many of the most recent messages survive verbatim.
        chat_model: Client used to generate the summary. Injected by tests;
            built from `settings` otherwise.
        settings: Configuration; defaults to the process-wide Settings.

    Returns:
        `messages` unchanged, with no model call made, if there are
        `keep_recent` or fewer of them — there is nothing to summarise.
        Otherwise, one `SystemMessage` carrying a summary of everything older,
        followed by the `keep_recent` most recent messages verbatim.
    """
    raise NotImplementedError("compaction — see the docstring above")


def drop_oldest(messages: Sequence[BaseMessage], *, budget: int) -> list[BaseMessage]:
    """Discard the oldest non-system messages until the transcript fits `budget`.

    Args:
        messages: The full conversation so far, oldest first.
        budget: Maximum total tokens the result may use.

    Returns:
        `messages` unchanged if already within `budget`. Otherwise, messages
        dropped oldest-first until the remainder fits. A leading `SystemMessage`
        is never dropped, even if the budget still is not met once every other
        message is gone.
    """
    raise NotImplementedError("compaction — see the docstring above")


class CompactingHistory(InMemoryChatMessageHistory):
    """A conversation's message history that keeps itself within a token budget.

    Storage — the `.messages` list and `.clear()` — is entirely inherited from
    `InMemoryChatMessageHistory`, not reimplemented. The only thing this class
    adds is deciding, every time a new message is written, whether the
    transcript has grown past `budget` and which strategy shrinks it if so —
    see `add_message`'s own docstring for exactly where that hook sits.

    Used directly — `history.add_messages([...])`, then read `history.messages`
    — not through `RunnableWithMessageHistory`. See the module docstring for
    why: that wrapper is deprecated, and its replacement is LangGraph.

    Attributes:
        budget: Token ceiling before compaction fires.
        strategy: `"summarise"`, `"drop_oldest"`, or `"none"`.
    """

    budget: int
    strategy: str = "summarise"

    _chat_model: BaseChatModel | None = PrivateAttr(default=None)

    def __init__(
        self,
        *,
        budget: int | None = None,
        strategy: str | None = None,
        chat_model: BaseChatModel | None = None,
        settings: Settings | None = None,
        **kwargs: Any,
    ) -> None:
        """Build a history bounded by `budget`, using `strategy` to enforce it.

        Args:
            budget: Token ceiling; defaults to `settings.history_token_budget`.
            strategy: Defaults to `settings.compaction_strategy`.
            chat_model: Forwarded to `summarise` on every compaction. Injected
                by tests; built from `settings` on first use otherwise.
            settings: Configuration; defaults to the process-wide Settings.
            **kwargs: Forwarded to `InMemoryChatMessageHistory` — `messages=`
                to seed a non-empty history, most likely.
        """
        settings = settings or get_settings()
        resolved_strategy = strategy or settings.compaction_strategy

        super().__init__(
            budget=settings.history_token_budget if budget is None else budget,
            strategy=resolved_strategy,
            **kwargs,
        )
        self._chat_model = chat_model

    def add_message(self, message: BaseMessage) -> None:
        """Store `message`, then compact if the transcript is now over budget.

        Overrides `add_message` (singular). The base class's `add_messages`
        (plural) already loops over a batch calling this method once per item,
        so a multi-message turn is still one logical write — overriding the
        singular form is simply the one place to hook in, not a name chosen to
        dodge anything.
        """
        raise NotImplementedError("history — see the docstring above")
