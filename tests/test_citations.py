"""`pipeline/citations.py` — the shape of a validated answer, and what catches a fake one."""

from __future__ import annotations

import pytest

from travel_assist.pipeline.citations import (
    Claim,
    DestinationAnswer,
    SelfCheckResult,
    UngroundedCitationError,
    self_check,
    validate_citations,
)


def test_a_claim_citing_an_included_chunk_passes():
    answer = DestinationAnswer(claims=[Claim(text="Porto has a bookshop.", source_chunk_id=1)])

    validate_citations(answer, {1, 2, 3})  # must not raise


def test_a_claim_citing_a_chunk_outside_the_context_is_caught():
    answer = DestinationAnswer(claims=[Claim(text="Porto has a bookshop.", source_chunk_id=999)])

    with pytest.raises(UngroundedCitationError, match="999"):
        validate_citations(answer, {1, 2, 3})


def test_a_claim_with_no_source_is_never_a_hallucination():
    """Not every claim needs a citation — a closing remark, say — only a false one is caught."""
    answer = DestinationAnswer(claims=[Claim(text="Let me know if you need anything else.")])

    validate_citations(answer, {1, 2, 3})  # must not raise


def test_every_hallucinated_chunk_id_is_named_not_just_the_first():
    answer = DestinationAnswer(
        claims=[
            Claim(text="A", source_chunk_id=1),
            Claim(text="B", source_chunk_id=888),
            Claim(text="C", source_chunk_id=999),
        ]
    )

    with pytest.raises(UngroundedCitationError, match=r"888.*999|999.*888"):
        validate_citations(answer, {1})


def test_an_answer_with_no_claims_at_all_passes():
    validate_citations(DestinationAnswer(claims=[]), {1, 2, 3})  # must not raise


def test_fraction_with_source_counts_only_sourced_claims():
    answer = DestinationAnswer(
        claims=[
            Claim(text="A", source_chunk_id=1),
            Claim(text="B", source_chunk_id=2),
            Claim(text="C"),
            Claim(text="D"),
        ]
    )

    assert answer.fraction_with_source == pytest.approx(0.5)


def test_fraction_with_source_is_zero_for_an_empty_answer():
    """Zero, not a division-by-zero error — an empty answer has no sourced claims."""
    assert DestinationAnswer(claims=[]).fraction_with_source == 0.0


# --------------------------------------------------------------------------
# self_check — missing citations, not wrong ones (that's validate_citations)
# --------------------------------------------------------------------------


def test_self_check_passes_when_every_claim_is_cited():
    answer = DestinationAnswer(
        claims=[
            Claim(text="A", source_chunk_id=1),
            Claim(text="B", source_chunk_id=2),
        ]
    )

    result = self_check(answer)

    assert result == SelfCheckResult(passed=True, uncited=[])


def test_self_check_fails_and_names_every_uncited_claim_in_order():
    cited = Claim(text="A", source_chunk_id=1)
    uncited_first = Claim(text="B")
    uncited_second = Claim(text="C")
    answer = DestinationAnswer(claims=[cited, uncited_first, uncited_second])

    result = self_check(answer)

    assert result.passed is False
    assert result.uncited == [uncited_first, uncited_second]


def test_self_check_passes_vacuously_on_an_answer_with_no_claims():
    """Documented deliberately: nothing uncited to find is not a failure here.

    An empty answer is `generate_answer`'s concern, not self-check's — see
    `self_check`'s own docstring.
    """
    result = self_check(DestinationAnswer(claims=[]))

    assert result == SelfCheckResult(passed=True, uncited=[])
