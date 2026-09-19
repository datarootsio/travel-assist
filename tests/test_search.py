"""`retrieval/search.py` — dense search, sparse search, and why neither is enough.

One file, because there is one exercise and one module. `vector_search` and
`keyword_search` share their SQL fragments, so the tests that describe the
*shape* of a result — ranked from one, scores descending, `k` bounds the count,
document metadata joined in — are parametrised over both functions rather than
written twice. What differs between them gets its own section, and the last
section is the argument for the milestone that follows.

Run against a real Postgres, never a fake. The behaviour being taught here *is*
Postgres behaviour — stemming, stop words, how `websearch_to_tsquery` parses what
a participant types — and a test double would assert my beliefs about full-text
search rather than what full-text search does. The embedding client is scripted
(see `retrieval_corpus.py`) so the geometry is a fact of the fixture instead of a
property of whichever model happens to be configured.
"""

from __future__ import annotations

import pytest

from travel_assist.config import Settings
from travel_assist.retrieval.models import RetrievedChunk, SearchFilters, SearchMethod
from travel_assist.retrieval.search import (
    hybrid_search,
    keyword_search,
    reciprocal_rank_fusion,
    vector_search,
)

from .retrieval_corpus import ScriptedEmbeddings, seed_corpus


@pytest.fixture
def corpus(clean_db: Settings) -> Settings:
    seed_corpus(clean_db)
    return clean_db


def vector(query: str, k: int, settings: Settings) -> list[RetrievedChunk]:
    """Dense search, with the fixture's hand-placed vectors in place of a provider."""
    return vector_search(query, k=k, settings=settings, embeddings=ScriptedEmbeddings())


def keyword(query: str, k: int, settings: Settings) -> list[RetrievedChunk]:
    """Sparse search. Same signature as `vector`, so both can be parametrised over."""
    return keyword_search(query, k=k, settings=settings)


def hybrid(
    query: str, k: int, settings: Settings, *, filters: SearchFilters | None = None
) -> list[RetrievedChunk]:
    """Hybrid search, with the fixture's scripted embeddings in place of a provider."""
    return hybrid_search(
        query, k=k, settings=settings, embeddings=ScriptedEmbeddings(), filters=filters
    )


def make_chunk(chunk_id: int, rank: int, method: SearchMethod = "vector") -> RetrievedChunk:
    """A minimal `RetrievedChunk` for RRF's own tests — no database involved.

    RRF only reads `chunk_id` and `rank`; every other field is a placeholder.
    """
    return RetrievedChunk(
        chunk_id=chunk_id,
        title="Somewhere",
        url="https://en.wikivoyage.org/wiki/Somewhere",
        content=f"chunk {chunk_id}",
        score=1.0,
        rank=rank,
        method=method,
    )


# A query each method matches several passages with, so ordering is observable.
BOTH_METHODS = pytest.mark.parametrize(
    ("search", "query"),
    [(vector, "where can I eat"), (keyword, "museums or beach or Shinkansen")],
    ids=["vector", "keyword"],
)


# --------------------------------------------------------------------------
# The result shape both methods promise
# --------------------------------------------------------------------------


@BOTH_METHODS
def test_results_are_ranked_and_numbered_from_one(search, query: str, corpus: Settings):
    results = search(query, 5, corpus)

    assert len(results) >= 3, "this query is meant to match several passages"
    assert [chunk.rank for chunk in results] == list(range(1, len(results) + 1))
    assert all(chunk.method == search.__name__ for chunk in results)


@BOTH_METHODS
def test_scores_are_descending(search, query: str, corpus: Settings):
    """`score` is a similarity for both methods — higher is better, never a distance."""
    scores = [chunk.score for chunk in search(query, 4, corpus)]

    assert len(scores) >= 3
    assert scores == sorted(scores, reverse=True)


@BOTH_METHODS
def test_k_bounds_the_result_count(search, query: str, corpus: Settings):
    assert len(search(query, 2, corpus)) == 2


@pytest.mark.parametrize(
    ("search", "query", "title", "country", "continent"),
    [
        (vector, "getting around by train", "Kyoto", "Japan", "Asia"),
        (keyword, "Rijksmuseum", "Amsterdam", "Netherlands", "Europe"),
    ],
    ids=["vector", "keyword"],
)
def test_document_metadata_is_joined_in(
    search, query: str, title: str, country: str, continent: str, corpus: Settings
):
    """Citations need the URL, and day 2's filters need country and continent."""
    top = search(query, 1, corpus)[0]

    assert top.title == title
    assert top.country == country
    assert top.continent == continent
    assert top.url.endswith(f"/{title}")


# --------------------------------------------------------------------------
# Dense search only
# --------------------------------------------------------------------------


def test_the_nearest_passage_comes_first_at_a_similarity_of_one(corpus: Settings):
    """An exact hit on a unit vector scores 1.0, and ranks first.

    The thing most likely to go wrong here is the SQL — specifically the
    operator, which has to match the index's operator class or Postgres quietly
    stops using the index and still returns the right answer.
    """
    results = vector("getting around by train", 3, corpus)

    assert "Shinkansen" in results[0].content
    assert results[0].score == pytest.approx(1.0)


def test_every_chunk_is_reachable_when_k_exceeds_the_corpus(corpus: Settings):
    results = vector("where can I eat", 50, corpus)

    assert len(results) == 10
    assert len({chunk.chunk_id for chunk in results}) == 10


# --------------------------------------------------------------------------
# Sparse search only — this is Postgres behaviour, not ours
# --------------------------------------------------------------------------


def test_stemming_means_a_different_word_form_still_matches(corpus: Settings):
    """`to_tsvector('english', ...)` stems, so "museums" and "museum" are one term."""
    singular = {chunk.chunk_id for chunk in keyword("museum", 10, corpus)}
    plural = {chunk.chunk_id for chunk in keyword("museums", 10, corpus)}

    assert singular & plural


def test_a_quoted_phrase_is_treated_as_a_phrase(corpus: Settings):
    """The reason for `websearch_to_tsquery` over `plainto_tsquery`.

    `plainto_tsquery` ANDs every word and ignores quotes entirely, so it cannot
    express "these words, in this order". Participants type quotes because every
    search box they have ever used supports them.
    """
    results = keyword('"open water bathing"', 5, corpus)

    assert results
    assert "open water bathing" in results[0].content.lower()


@pytest.mark.parametrize(
    "query", ["!!!", "the of and", "   "], ids=["unparseable", "only-stop-words", "empty"]
)
def test_a_query_with_no_search_terms_returns_nothing_rather_than_raising(
    query: str, corpus: Settings
):
    """User input reaches this function directly. It must not be able to crash it."""
    assert keyword(query, 5, corpus) == []


# --------------------------------------------------------------------------
# The paired failure — the argument for hybrid search
# --------------------------------------------------------------------------
#
# Two queries, two methods, and each method fails the query the other answers:
#
#     "somewhere to swim outdoors"   vector finds it   keyword returns nothing
#     "Jan Tooropstraat"             keyword finds it  vector ranks it nowhere
#
# Neither failure is a bug. Sparse retrieval cannot match a word that is not
# there, and dense retrieval smooths a rare proper noun into the topic it
# resembles. Nothing in either implementation can be tuned to fix this, which is
# exactly why the next milestone stops choosing between them and fuses both.
#
# These are regression tests for the *teaching argument*, and they are
# deliberately implementation-coupled in the same spirit as `test_seam.py`: if a
# later change makes one of these queries succeed under both methods, hybrid
# search has lost its motivating example and the guide needs rewriting. That is
# worth being told about by a failing test.
#
# The demonstration for the room is `travel-assist search` against the real
# index. This only stops it silently rotting.

SEMANTIC_QUERY = "somewhere to swim outdoors"
PROPER_NOUN_QUERY = "Jan Tooropstraat"


def test_keyword_cannot_answer_a_paraphrased_question(corpus: Settings):
    """The passage is in the corpus. None of the query's words are."""
    assert keyword(SEMANTIC_QUERY, 8, corpus) == []


def test_vector_answers_it(corpus: Settings):
    """The whole point of dense retrieval: the top hit shares no word with the query."""
    top = vector(SEMANTIC_QUERY, 3, corpus)[0].content.lower()

    assert "open water bathing" in top
    assert not {"swim", "outdoors"} & set(top.split()), "this query should share no term"


def test_keyword_answers_an_exact_proper_noun(corpus: Settings):
    results = keyword(PROPER_NOUN_QUERY, 8, corpus)

    assert results
    assert PROPER_NOUN_QUERY in results[0].content


def test_vector_misses_it_and_returns_topically_similar_passages_instead(corpus: Settings):
    """Not noise — plausible, related, wrong. The failure mode that costs trust."""
    results = vector(PROPER_NOUN_QUERY, 3, corpus)

    assert results, "the failure is a wrong answer, not an empty one"
    assert not any(PROPER_NOUN_QUERY in chunk.content for chunk in results)
    # What it returns instead is things to go and see. Someone looking for a
    # hospital address is handed a shrine.
    assert "Fushimi Inari" in results[0].content


def test_neither_method_answers_both_queries(corpus: Settings):
    """The claim hybrid search exists to resolve, in one assertion.

    Written as a property rather than two examples so it keeps its meaning if
    the fixture corpus grows: for each method there is a query it cannot answer.
    """
    answered_by_keyword = {
        query for query in (SEMANTIC_QUERY, PROPER_NOUN_QUERY) if keyword(query, 8, corpus)
    }
    answered_by_vector = {
        query
        for query in (SEMANTIC_QUERY, PROPER_NOUN_QUERY)
        if any(
            PROPER_NOUN_QUERY in chunk.content or "bathing" in chunk.content
            for chunk in vector(query, 3, corpus)
        )
    }

    assert answered_by_keyword == {PROPER_NOUN_QUERY}
    assert answered_by_vector == {SEMANTIC_QUERY}


# --------------------------------------------------------------------------
# Reciprocal Rank Fusion — pure function, no database
# --------------------------------------------------------------------------
#
# Expected values are hand-computed, not recomputed the way the code computes
# them (the tautology to avoid when testing): a fused score is a sum of
# 1 / (k_rrf + rank) terms, worked out here with a calculator, not by re-deriving
# the formula in the assertion.


def test_a_chunk_ranked_well_in_two_rankings_beats_one_ranked_first_in_only_one():
    """The property that makes fusion worth having, not just a merge.

    Ranking A: [1, 2, 3].  Ranking B: [2, 3, 1].  With k_rrf=60:
        chunk 1: 1/61 + 1/63 = 0.03226...
        chunk 2: 1/62 + 1/61 = 0.03252...   <- highest: 2nd and 1st
        chunk 3: 1/63 + 1/62 = 0.03200...
    Chunk 2 is never first in either ranking, but it is consistently near the
    top of both, and that beats chunk 1's single first place.
    """
    ranking_a = [make_chunk(1, 1), make_chunk(2, 2), make_chunk(3, 3)]
    ranking_b = [make_chunk(2, 1), make_chunk(3, 2), make_chunk(1, 3)]

    fused = reciprocal_rank_fusion([ranking_a, ranking_b])

    assert [result.chunk_id for result in fused] == [2, 1, 3]
    assert fused[0].score == pytest.approx(1 / 62 + 1 / 61)
    assert fused[1].score == pytest.approx(1 / 61 + 1 / 63)
    assert fused[2].score == pytest.approx(1 / 63 + 1 / 62)


def test_fused_results_are_ranked_from_one_and_tagged_hybrid():
    fused = reciprocal_rank_fusion([[make_chunk(1, 1)], [make_chunk(2, 1)]])

    assert [result.rank for result in fused] == [1, 2]
    assert all(result.method == "hybrid" for result in fused)


def test_a_chunk_missing_from_one_ranking_still_gets_a_score_from_the_other():
    """A method that finds nothing must not veto a chunk the other method found."""
    fused = reciprocal_rank_fusion([[make_chunk(1, 1)], []])

    assert [result.chunk_id for result in fused] == [1]
    assert fused[0].score == pytest.approx(1 / 61)


def test_a_chunk_present_in_both_rankings_is_not_duplicated():
    fused = reciprocal_rank_fusion([[make_chunk(1, 1)], [make_chunk(1, 1)]])

    assert [result.chunk_id for result in fused] == [1]


def test_no_rankings_fuse_to_an_empty_list():
    assert reciprocal_rank_fusion([]) == []
    assert reciprocal_rank_fusion([[], []]) == []


def test_k_rrf_actually_changes_the_fused_score():
    """A guard against the constant being accepted but silently ignored.

    Same two rankings as the first test above, fused with k_rrf=1 instead of the
    default 60: chunk 1 scores 1/(1+1)=0.5, chunk 2 scores 1/(1+2)+1/(1+1)=0.833.
    Both the values and the *size of the gap* between them must move with
    k_rrf — a hardcoded 60 would still pass every other test in this file.
    """
    ranking_a = [make_chunk(1, 1), make_chunk(2, 2)]
    ranking_b = [make_chunk(2, 1)]

    fused = reciprocal_rank_fusion([ranking_a, ranking_b], k_rrf=1)

    assert [result.chunk_id for result in fused] == [2, 1]
    assert fused[0].score == pytest.approx(1 / 3 + 1 / 2)
    assert fused[1].score == pytest.approx(1 / 2)


# --------------------------------------------------------------------------
# hybrid_search — the fusion of the two, against the fixture corpus
# --------------------------------------------------------------------------


def test_hybrid_results_are_ranked_from_one_and_tagged_hybrid(corpus: Settings):
    results = hybrid("where can I eat", 5, corpus)

    assert len(results) >= 3
    assert [result.rank for result in results] == list(range(1, len(results) + 1))
    assert all(result.method == "hybrid" for result in results)


def test_k_bounds_the_fused_result_count(corpus: Settings):
    assert len(hybrid("where can I eat", 2, corpus)) == 2


def test_hybrid_recovers_the_semantic_failure(corpus: Settings):
    """Fusion still surfaces the passage keyword search alone cannot find.

    Keyword alone returns nothing for this query (see above); vector alone
    already ranked the passage first, and fusing with an empty ranking cannot
    make that worse.
    """
    results = hybrid(SEMANTIC_QUERY, 5, corpus)

    assert results
    assert "open water bathing" in results[0].content


def test_hybrid_recovers_the_proper_noun_failure(corpus: Settings):
    """Fusion recovers the passage vector search alone buries.

    Vector alone buries this at rank 10 of 10 (see above); keyword alone finds
    it at rank 1. Fused, the chunk that appears — even weakly — in both
    rankings outranks every chunk that only vector likes, exactly like the RRF
    unit test above: present-in-both beats first-in-one.
    """
    results = hybrid(PROPER_NOUN_QUERY, 5, corpus)

    assert results
    assert PROPER_NOUN_QUERY in results[0].content


def test_filters_narrow_results_to_matching_documents(corpus: Settings):
    results = hybrid("where can I eat", 8, corpus, filters=SearchFilters(country="Portugal"))

    assert results
    assert all(result.country == "Portugal" for result in results)


def test_filters_can_exclude_every_result(corpus: Settings):
    """A contradictory filter narrows to nothing rather than raising or being ignored.

    No document in the fixture is both Portuguese and Asian.
    """
    results = hybrid(
        "where can I eat", 8, corpus, filters=SearchFilters(country="Portugal", continent="Asia")
    )

    assert results == []
