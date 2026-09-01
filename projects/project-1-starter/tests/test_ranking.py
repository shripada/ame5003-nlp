"""Stage 4. The arithmetic first, then what it does to real results."""

import math

import pytest

from minisearch import evaluate, ranking
from minisearch.index import build_index
from minisearch.pipeline import Analyzer


@pytest.mark.parametrize("count,expected", [(1, 1.0), (10, 2.0), (100, 3.0), (1000, 4.0)])
def test_tf_weight_is_one_plus_log10(count, expected):
    assert ranking.tf_weight(count) == pytest.approx(expected)


def test_tf_weight_has_diminishing_returns():
    """Session 10's reason for the first logarithm: 1 to 10 counts for as much
    as 10 to 100, because relevance does not grow linearly with the count."""
    assert (ranking.tf_weight(10) - ranking.tf_weight(1)) == \
        pytest.approx(ranking.tf_weight(100) - ranking.tf_weight(10))


def test_a_term_in_every_document_has_zero_idf(toy_docs, analyzer):
    """The claim session 10 makes about stop words, tested directly. A term
    added to every document must contribute nothing to any score."""
    everywhere = [doc + "\nubiquitousterm" for doc in toy_docs]
    index = build_index(everywhere, analyzer)
    term = analyzer.terms("ubiquitousterm")[0]
    assert index.df(term) == index.n_docs
    assert ranking.idf(index, term) == pytest.approx(0.0)


def test_idf_falls_as_df_rises(toy_index, analyzer):
    cat, bird = (analyzer.terms(w)[0] for w in ("cat", "bird"))
    assert toy_index.df(cat) > toy_index.df(bird)
    assert ranking.idf(toy_index, cat) < ranking.idf(toy_index, bird)


def test_idf_of_an_unknown_term_is_zero_not_an_exception(toy_index):
    assert ranking.idf(toy_index, "zzzzzz") == 0.0


def test_idf_is_base_ten(toy_index, analyzer):
    bird = analyzer.terms("bird")[0]
    expected = math.log10(toy_index.n_docs / toy_index.df(bird))
    assert ranking.idf(toy_index, bird) == pytest.approx(expected), \
        "base e changes every number in your report"


def test_results_are_sorted_and_capped(toy_index):
    results = ranking.search_ranked(toy_index, "cat dog bird", k=3)
    assert len(results) <= 3
    assert [score for _, score in results] == sorted((s for _, s in results), reverse=True)


def test_only_candidate_documents_are_scored(toy_index, analyzer):
    """Score the documents on the query terms' postings lists and no others.

    Walking all of them gives the same ranking by the wrong algorithm, and on
    the full corpus it is a hundred times slower.
    """
    candidates = set(toy_index.postings(analyzer.terms("bird")[0]))
    scored = {doc_id for doc_id, _ in ranking.search_ranked(toy_index, "bird", k=10)}
    assert scored == candidates


def test_ties_are_broken_by_document_id(analyzer):
    """Two documents with identical scores must come back in document order.

    Without a secondary sort key the order depends on dictionary insertion, and
    two students running the same code quote different top-tens.
    """
    # Four identical documents, and a fifth that shares no term — without the
    # fifth, `cotton` would be in every document, its idf would be 0, and there
    # would be nothing to rank.
    docs = ["cotton harvest report"] * 4 + ["shipping freight tonnage"]
    index = build_index(docs, analyzer)
    results = ranking.search_ranked(index, "cotton harvest", k=4)
    assert len({score for _, score in results}) == 1, "this test is pointless unless they tie"
    assert [doc_id for doc_id, _ in results] == [0, 1, 2, 3]


def test_n_matching_counts_every_document_that_can_score(toy_index, analyzer):
    """The M in "10 results of 328" — the union of the query terms' postings."""
    cat, bird = (analyzer.terms(w)[0] for w in ("cat", "bird"))
    expected = len(set(toy_index.postings(cat)) | set(toy_index.postings(bird)))
    assert ranking.n_matching(toy_index, "cat bird") == expected
    assert ranking.n_matching(toy_index, "zzzzzz") == 0


def test_n_matching_is_at_least_the_number_returned(toy_index):
    results = ranking.search_ranked(toy_index, "cat dog bird", k=2)
    assert ranking.n_matching(toy_index, "cat dog bird") >= len(results)


def test_every_returned_score_is_positive(toy_index):
    assert all(score > 0 for _, score in ranking.search_ranked(toy_index, "cat", k=10))


def test_an_unknown_query_returns_nothing(toy_index):
    assert ranking.search_ranked(toy_index, "zzzzzz", k=10) == []


def test_the_worked_example_from_session_10():
    """Session 10's exercise, run as a test.

    Two documents match `car insurance`. A contains `insurance` nine times; B
    contains `the` fifty times and `insurance` once. A must rank first, and `the`
    must contribute nothing to either — it is in every document, so its idf is
    exactly zero. That comparison is the whole behaviour of TF-IDF.
    """
    docs = [
        "the car insurance " + "insurance " * 8,      # A
        "the car " + "the " * 49 + "insurance",       # B
        "the market report",
        "the weather today",
    ]
    plain = Analyzer(fold=True, drop_stop=False, stem=False)
    index = build_index(docs, plain)

    assert ranking.idf(index, "the") == pytest.approx(0.0), \
        "`the` is in every document here, so idf must be exactly 0"

    top = ranking.search_ranked(index, "car insurance", k=4)
    assert top[0][0] == 0, "the document actually concerned with insurance must win"

    with_the = ranking.search_ranked(index, "the car insurance", k=4)
    assert [doc_id for doc_id, _ in with_the] == [doc_id for doc_id, _ in top], \
        "adding a zero-idf word to the query must not change the ranking"


@pytest.mark.slow
def test_the_corpus_idf_values(reuters_index, analyzer):
    """The numbers the project brief quotes. If yours differ, something upstream
    changed the terms, not the arithmetic."""
    oil, opec, said = (analyzer.terms(w)[0] for w in ("oil", "opec", "said"))
    assert ranking.idf(reuters_index, oil) == pytest.approx(1.006, abs=0.01)
    assert ranking.idf(reuters_index, opec) == pytest.approx(1.950, abs=0.01)
    assert ranking.idf(reuters_index, said) == pytest.approx(0.201, abs=0.01)


@pytest.mark.slow
def test_precision_on_the_supplied_queries(reuters_index):
    def search(query, k):
        return ranking.search_ranked(reuters_index, query, k)

    mean = evaluate.precision_at_k(search, k=10)
    assert mean >= 0.80, (
        f"mean P@10 is {mean:.2f}. A correct pipeline reaches about 0.90; below 0.6 the "
        "usual cause is the query being analysed differently from the index."
    )


@pytest.mark.slow
def test_queries_are_fast(reuters_index):
    """A query that takes longer than this is scoring every document."""
    import time

    query = "interest rate cut by the federal reserve"
    ranking.search_ranked(reuters_index, query, k=10)      # warm any caches
    t0 = time.perf_counter()
    for _ in range(20):
        ranking.search_ranked(reuters_index, query, k=10)
    ms = (time.perf_counter() - t0) / 20 * 1000
    assert ms < 200, f"{ms:.0f} ms per query — you are almost certainly scoring every document"
