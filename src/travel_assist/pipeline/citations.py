"""What a validated answer looks like, and the check that keeps it honest.

Diff from hybrid-search: the model no longer returns free text. It returns
`DestinationAnswer` — a list of claims, each naming the `chunk_id` of the passage
that supports it, or none if it cannot ground the claim. Free text can assert
anything; a claim that must name its source can be checked, and `validate_citations`
is the check — it catches a claim that cites a chunk_id the model was never shown,
which is what "grounded" actually has to mean here: not "sounds like it came from
the corpus," but "provably did."
"""

from __future__ import annotations

from collections.abc import Collection

from pydantic import BaseModel


class Claim(BaseModel):
    """One assertion in an answer, and the chunk that supports it, if any.

    Attributes:
        text: The assertion itself, in the model's own words.
        source_chunk_id: The `chunk_id` of the passage that supports `text`, or
            `None` if the model could not ground this claim in the retrieved
            context. A claim with no source is not an error by itself — a
            closing remark or a clarifying question is a legitimate claim with
            nothing to cite — but a source that names a chunk outside the
            assembled context is, and `validate_citations` is what catches that.
    """

    text: str
    source_chunk_id: int | None = None


class DestinationAnswer(BaseModel):
    """A structured, checkable answer instead of a free-text blob.

    Attributes:
        claims: Every assertion in the answer, in the order they should be read.
    """

    claims: list[Claim]

    @property
    def fraction_with_source(self) -> float:
        """The fraction of claims carrying a `source_chunk_id` — the prompt-comparison number.

        `0.0` for an answer with no claims at all, rather than raising — an
        empty answer has no sourced claims, which is a fact about it, not an
        error dividing by zero should mask.
        """
        if not self.claims:
            return 0.0
        return sum(1 for claim in self.claims if claim.source_chunk_id is not None) / len(
            self.claims
        )


class UngroundedCitationError(ValueError):
    """A claim cited a chunk_id that was never in the assembled context.

    Raised instead of silently trusting the model: an answer that cites its
    passages is only grounded if the citations are real, and a model asked to
    cite sources will sometimes invent a plausible-looking one rather than admit
    it does not know.
    """


def validate_citations(answer: DestinationAnswer, included_chunk_ids: Collection[int]) -> None:
    """Assert every cited chunk_id was actually in the context the model was shown.

    Args:
        answer: The model's structured response.
        included_chunk_ids: Every `chunk_id` that was in the prompt — typically
            `{chunk.chunk_id for chunk in context}` for whatever `context` was
            passed to the call that produced `answer`.

    Raises:
        UngroundedCitationError: If any claim's `source_chunk_id` is set but is
            not a member of `included_chunk_ids`.
    """
    raise NotImplementedError("structured-answer — see the docstring above")


class SelfCheckResult(BaseModel):
    """Whether every claim in an answer carries a citation, and which do not.

    A cheaper, narrower check than `validate_citations`: it never looks at
    what was in the assembled context, so it costs nothing but the answer
    itself.

    Attributes:
        passed: True iff every claim has a `source_chunk_id`.
        uncited: The claims with none, in the order they appear in the answer.
            Empty when `passed`.
    """

    passed: bool
    uncited: list[Claim]


def self_check(answer: DestinationAnswer) -> SelfCheckResult:
    """Check that `answer` leaves no claim uncited.

    Diff from `validate_citations`: that function catches a claim citing a
    chunk_id that was never shown to the model — a citation that is *wrong*.
    This one catches a claim with no citation at all — a citation that is
    *missing*. An answer can fail either independently of the other, which is
    why the day-1 pipeline runs both (`pipeline/chain.py`).

    Args:
        answer: The model's structured response.

    Returns:
        A `SelfCheckResult` naming every uncited claim, if any. `passed` is
        `True` on an answer with no claims at all — there is nothing uncited
        to find, which is a fact about the answer, not a reason to fail it
        here; an empty answer is `generate_answer`'s concern, not self-check's.
    """
    raise NotImplementedError("self-check — see the docstring above")
