"""Prompt text for turning retrieved chunks into a cited answer.

Plain Python strings, not a template engine and not a registry:
prompts live in Python, next to the code that uses them.

`ANSWER_PROMPT` and `ANSWER_PROMPT_NO_SOURCE_REQUIREMENT` differ in exactly one
instruction — demanding a source per claim, or not — and nothing else. That
narrowness is deliberate: the prompt comparison in `pipeline/steps.py` runs both
over the same questions and reports one number, the fraction of claims carrying a
source. If the two prompts differed in more than that one instruction, the number
would be measuring the wrong thing.

Both carry the same **scope-check** paragraph, which is narrower than
`pipeline/routing.py`'s `classify_route`: routing refuses a whole out-of-scope
*question* before retrieval ever runs; this catches the case where an in-scope
question's retrieval still returns a chunk about the wrong destination — routing
only ever looks at the question text, never at what retrieval actually returns.
"""

from __future__ import annotations

_SCOPE_CHECK = """Retrieval returns close matches even when nothing relevant exists, \
so a passage below may be about the wrong destination. If one is, say so \
explicitly rather than answering as if it were right. If the passages match, \
say nothing about scope — do not add a claim confirming that they do."""

ANSWER_PROMPT = f"""You are a travel assistant answering questions about destinations \
using ONLY the passages below.

{_SCOPE_CHECK}

Every factual claim you make MUST cite the chunk_id of the passage that supports \
it. If a claim cannot be supported by any passage below, either omit it or say so \
explicitly — never cite a chunk_id whose passage does not actually contain the \
information.

Passages:
{{context}}

Question: {{query}}

Answer as a list of claims, each naming the chunk_id that supports it."""

ANSWER_PROMPT_NO_SOURCE_REQUIREMENT = f"""You are a travel assistant answering \
questions about destinations using the passages below.

{_SCOPE_CHECK}

Passages:
{{context}}

Question: {{query}}

Answer as a list of claims."""
