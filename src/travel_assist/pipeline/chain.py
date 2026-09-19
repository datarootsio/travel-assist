"""Diff from routing/self-check: every step lands in one runnable.

Routing, `hybrid_search`, context assembly, the cited answer, and self-check
all compose into one call, with a `RunnableBranch` deciding first whether
retrieval even runs. This is the whole day-1 pipeline, and it is what day 2
takes apart: the same steps become tools, and the model — not this function —
decides their order and how many times each runs.

The behaviour that matters most here is not what an answer looks like, it is a
branch that is never reached: `out_of_scope` and `needs_clarification` never
call `hybrid_search`. Nothing catches a wrongly-answered out-of-scope question
after the fact — the routing decision happens first, and retrieval simply does
not run on those branches. `PipelineTurn.retrieved` records this directly, and
`tests/test_chain.py` proves it the harder way too: by spying on
`hybrid_search` itself and asserting it was never called.

**Multi-turn coreference** goes through `condense_question`: a real
`prompt | model | StrOutputParser()` LCEL chain — this repo's first — that
rewrites a follow-up like "what about the food there?" into a standalone
question ("What food does Porto have?") given the prior turns as actual
messages, not string-glued. That standalone question is what `hybrid_search`
and `classify_route` see. `generate_answer` gets something different again:
the plain, unrewritten `query`, plus `history` as its own parameter — real
message objects the model reads as conversation, rather than baked into the
question text. Nothing here is `langchain.chains.create_history_aware_retriever`
— that module does not exist in the installed `langchain`; this is the
hand-built equivalent.

A compound question naming two destinations still only reaches `hybrid_search`
once, because this pipeline retrieves exactly once by construction.
`hybrid_search`'s default `k=8`, fused across two destinations' worth of
chunks, is thin next to a dedicated search per destination, and the result is
a demonstrably incomplete comparison. That is not fixed here —
it is day 2's opening motivation, where retrieval becomes something the
model can call more than once.
"""

from __future__ import annotations

from collections.abc import Sequence

from langchain_core.language_models import BaseChatModel, LanguageModelInput
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import Runnable, RunnableBranch, RunnableConfig, RunnableLambda
from pydantic import BaseModel

from travel_assist.config import Settings, get_settings
from travel_assist.db import list_corpus_destinations
from travel_assist.models import get_chat_model
from travel_assist.pipeline.citations import DestinationAnswer, SelfCheckResult, self_check
from travel_assist.pipeline.context import assemble_context
from travel_assist.pipeline.memory import CompactingHistory
from travel_assist.pipeline.routing import Route, RouteDecision, classify_route
from travel_assist.pipeline.steps import generate_answer
from travel_assist.retrieval.models import RetrievedChunk
from travel_assist.retrieval.search import hybrid_search

_CONDENSE_INSTRUCTION = (
    "Given the conversation so far and a follow-up question, rewrite the "
    "follow-up as a standalone question answerable without the conversation. "
    "Do not answer it — only rewrite it. If it is already standalone, return "
    "it unchanged. Return only the rewritten question, nothing else."
)

_CONDENSE_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", _CONDENSE_INSTRUCTION),
        MessagesPlaceholder("history"),
        ("human", "{query}"),
    ]
)

_CLARIFYING_QUESTION = (
    "I can help with that once I know which destination you mean. I can answer "
    "questions about: {destinations}."
)


class PipelineTurn(BaseModel):
    """One turn's outcome: which route was taken, and what came of it.

    Attributes:
        route: What `classify_route` decided.
        retrieved: True only on the `destination_question` branch. False for
            `out_of_scope` and `needs_clarification`, which never call
            `hybrid_search` — the property "out-of-scope refuses without
            retrieving" reduces to this field.
        answer: Set only when `route == "destination_question"`.
        self_check: `self_check`'s verdict on `answer`. Set only alongside it.
        context: The chunks `answer` was allowed to cite. Empty on the other
            two branches.
        message: The refusal (`out_of_scope`) or the clarifying question
            (`needs_clarification`). Never set alongside `answer`.
    """

    route: Route
    retrieved: bool = False
    answer: DestinationAnswer | None = None
    self_check: SelfCheckResult | None = None
    context: list[RetrievedChunk] = []
    message: str | None = None

    @property
    def display_text(self) -> str:
        """What a CLI or a stored history message should show for this turn."""
        if self.message is not None:
            return self.message
        assert self.answer is not None
        return " ".join(claim.text for claim in self.answer.claims)


def _display_names(destinations: Sequence[str]) -> list[str]:
    """Collapse `"Lisbon/Alfama"`-style district articles down to their city.

    `destinations` is one `documents.title` per article — Lisbon's 6 district
    pages are separate rows from `"Lisbon"` itself, so the raw list has 16
    entries for what a traveller would call 10 destinations. `classify_route`
    gets the full, un-collapsed list — the extra specificity can only help it
    recognise a district-level question — but a refusal or clarifying message
    shown to a person should read as "10 destinations", not 16 near-duplicates.
    """
    seen: list[str] = []
    for name in destinations:
        city = name.split("/", 1)[0]
        if city not in seen:
            seen.append(city)
    return seen


def condense_question(
    query: str,
    history: Sequence[BaseMessage],
    *,
    chat_model: BaseChatModel | None = None,
    settings: Settings | None = None,
) -> str:
    """Rewrite `query` into a standalone question, given prior turns.

    Used for retrieval and routing, both of which need one self-contained
    question to work from — neither reads a message history.

    Args:
        query: The user's new question, exactly as typed.
        history: Prior turns, oldest first, read as real messages rather than
            flattened into text.
        chat_model: Client used to rewrite the question. Injected by tests;
            built from `settings` otherwise.
        settings: Configuration; defaults to the process-wide Settings.

    Returns:
        `query` unchanged, with no model call made, when `history` is empty —
        there is nothing yet for a reference like "there" to resolve against.
        Otherwise, the model's standalone rewrite.
    """
    raise NotImplementedError("routing — see the docstring above")


def answer_query(
    query: str,
    *,
    history: CompactingHistory | None = None,
    settings: Settings | None = None,
    route_model: Runnable[LanguageModelInput, RouteDecision] | None = None,
    structured_model: Runnable[LanguageModelInput, DestinationAnswer] | None = None,
    condense_chat_model: BaseChatModel | None = None,
    destinations: Sequence[str] | None = None,
    config: RunnableConfig | None = None,
) -> PipelineTurn:
    """Route `query`, then answer, refuse, or ask for clarification — one call.

    Args:
        query: The user's raw question. Stored into `history` unchanged, and
            what `generate_answer` sees — never rewritten.
        history: Prior turns. Read by `condense_question` (to resolve routing
            and retrieval) and passed to `generate_answer` as real messages;
            appended to after this turn. `None` for a single-shot call with
            no memory.
        settings: Configuration; defaults to the process-wide Settings.
        route_model: Forwarded to `classify_route`. Injected by tests.
        structured_model: Forwarded to `generate_answer`. Injected by tests.
        condense_chat_model: Forwarded to `condense_question`. Injected by tests.
        destinations: The corpus's known destinations, one per indexed
            article. Injected by tests; queried fresh from the index
            otherwise. Queried once, then reused for both routing (the full
            list) and any refusal/clarification message (collapsed to one
            entry per city — see `_display_names`).
        config: Trace tags for this call, from
            `observability.tracing.traced_config`. `None` runs untraced.

    Returns:
        A `PipelineTurn` naming the route taken and what came of it.
    """
    raise NotImplementedError("routing — see the docstring above")
