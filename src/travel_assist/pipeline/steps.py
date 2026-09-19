"""One day-1 step: retrieved, budgeted chunks become a validated, cited answer.

Diff from context-budget: assembly decided *which* chunks reach the prompt; this
decides what the model is allowed to say about them. The model returns
`DestinationAnswer` — a list of claims, each citing the `chunk_id` that supports
it — never free text, and `validate_citations` catches a claim that cites a
chunk_id outside what was actually in the prompt. See `pipeline/citations.py`.

`generate_answer` is what the prompt-comparison exercise calls twice, once per
prompt in `pipeline/prompts.py`, over the same questions.

`history` is sent as real messages ahead of the formatted prompt, so multi-turn
coreference is the model's own job, not something baked into `query`
beforehand. See `pipeline/chain.py::condense_question` for the retrieval-side
counterpart, which rewrites `query` instead.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import cast

from langchain_core.language_models import LanguageModelInput
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.runnables import Runnable

from travel_assist.config import Settings, get_settings
from travel_assist.models import get_chat_model
from travel_assist.pipeline.citations import DestinationAnswer, validate_citations
from travel_assist.pipeline.prompts import ANSWER_PROMPT
from travel_assist.retrieval.models import RetrievedChunk


def _format_context(chunks: Sequence[RetrievedChunk]) -> str:
    """Render assembled chunks so the model can cite them by `chunk_id`."""
    return "\n\n".join(
        f"[chunk_id={chunk.chunk_id}] {chunk.citation_label()}\n{chunk.content}" for chunk in chunks
    )


def generate_answer(
    query: str,
    context: Sequence[RetrievedChunk],
    *,
    history: Sequence[BaseMessage] = (),
    prompt_template: str = ANSWER_PROMPT,
    settings: Settings | None = None,
    structured_model: Runnable[LanguageModelInput, DestinationAnswer] | None = None,
) -> DestinationAnswer:
    """Answer `query` from `context` alone, as a validated, cited `DestinationAnswer`.

    Args:
        query: The user's question, forwarded into the prompt unchanged — not
            rewritten, even when `history` is non-empty. A reference like
            "there" is left for the model to resolve using `history`, the way
            any multi-turn chat model would.
        context: Chunks to answer from — typically `assemble_context`'s output.
            Citations outside this set are rejected, so pass exactly what the
            prompt is allowed to draw on.
        history: Prior turns, oldest first, sent as real messages ahead of the
            formatted prompt — not folded into `query` or `context`. Empty by
            default: a single-shot call sends only the formatted prompt.
        prompt_template: The instruction, as a `{query}`/`{context}` format
            string. Swappable for the prompt-comparison exercise; see
            `pipeline/prompts.py`.
        settings: Configuration; defaults to the process-wide Settings.
        structured_model: A `Runnable` returning `DestinationAnswer`, used
            instead of building one from `settings`. Injected by tests with a
            scripted double; built from `get_chat_model()` otherwise.

    Returns:
        A `DestinationAnswer` whose every citation names a `chunk_id` present in
        `context`.

    Raises:
        UngroundedCitationError: If the model cites a chunk_id not in `context`.
    """
    raise NotImplementedError("structured-answer — see the docstring above")
