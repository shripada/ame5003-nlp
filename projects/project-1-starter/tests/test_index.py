"""Stage 3 — the merges, then the index they run over."""

import pytest

from minisearch.index import build_index, difference, intersect, union
from minisearch.matrix import build_matrix, projected_size
from minisearch.pipeline import Analyzer

A = [1, 3, 5, 7, 9]
B = [3, 4, 5, 9, 11]


def test_intersect():
    assert intersect(A, B) == [3, 5, 9]


def test_union():
    assert union(A, B) == [1, 3, 4, 5, 7, 9, 11]


def test_difference():
    assert difference(A, B) == [1, 7]


@pytest.mark.parametrize("merge", [intersect, union, difference])
def test_merges_handle_an_empty_side(merge):
    assert merge([], B) == ([] if merge is not union else B)
    assert merge(A, []) == ([] if merge is intersect else A)


@pytest.mark.parametrize("merge", [intersect, union, difference])
def test_merges_return_sorted_unique_lists(merge):
    out = merge(A, B)
    assert out == sorted(set(out))
    assert isinstance(out, list)


def test_merges_do_not_mutate_their_arguments():
    a, b = list(A), list(B)
    intersect(a, b), union(a, b), difference(a, b)
    assert (a, b) == (A, B)


def test_intersect_is_a_merge_not_a_set_intersection():
    """A merge walks two sorted lists once. `set(p1) & set(p2)` gives the same
    answer, but session 9 was about the merge, so the merge is what is marked.

    Unsorted input is not something a postings list can be, so a real merge is
    free to give the wrong answer here — and a set intersection cannot.
    """
    assert intersect([5, 1, 3], [3, 1, 5]) != [1, 3, 5], \
        "this looks like set(p1) & set(p2) rather than a linear merge"


# ── the index itself ─────────────────────────────────────────────────────

def test_index_covers_every_document(toy_index, toy_docs):
    assert toy_index.n_docs == len(toy_docs)
    assert toy_index.vocabulary_size() > 0


def test_postings_are_sorted_and_unique(toy_index):
    for term in toy_index.entries:
        docs = toy_index.postings(term)
        assert docs == sorted(set(docs)), f"postings list for {term!r} is not sorted and unique"


def test_df_is_the_length_of_the_postings_list(toy_index):
    for term in toy_index.entries:
        assert toy_index.df(term) == len(toy_index.postings(term))


def test_counts_are_stored(toy_index, analyzer):
    """Stage 4 needs them, and adding them later means rebuilding."""
    term = analyzer.terms("dog")[0]
    counts = dict(toy_index.counts(term))
    assert counts[1] == 3, "document 1 says 'dog' three times, headline included"


def test_the_toy_postings(toy_index, analyzer):
    cat, dog, bird = (analyzer.terms(w)[0] for w in ("cat", "dog", "bird"))
    assert toy_index.postings(cat) == [0, 2, 5]
    assert toy_index.postings(dog) == [0, 1, 2]
    assert toy_index.postings(bird) == [3, 5]
    assert intersect(toy_index.postings(cat), toy_index.postings(bird)) == [5]
    assert difference(toy_index.postings(cat), toy_index.postings(dog)) == [5]


def test_lab_4s_own_collection(lab4_index):
    """The continuity check: lab 4's six documents, lab 4's answers.

    Lab 4 numbered its documents from 1 and this project numbers them from 0, so
    every ID here is one lower than the one on the board. Nothing else differs:
    `cat` was 1, 3, 6 there and is 0, 2, 5 here, the vocabulary was 23 and is 23,
    and `cat AND bird` was document 6 and is document 5.
    """
    assert lab4_index.vocabulary_size() == 23
    assert lab4_index.postings("cat") == [0, 2, 5]
    assert lab4_index.postings("dog") == [1, 2]
    assert lab4_index.postings("bird") == [4, 5]
    assert lab4_index.boolean_and("cat bird") == [5]
    assert lab4_index.boolean_and_not("cat", "dog") == [0, 5]


def test_stemming_changes_lab_4s_answer(lab4_index, lab4_docs):
    """The same six documents, stemmed, do not give lab 4's answer — and that is
    the point of sessions 4 and 5 rather than a bug.

    Lab 4 tokenised with a regex and no stemming, so `cats` in document 4 was a
    different term from `cat`. Under this project's default analyser they are one
    term, and document 3 (lab 4's document 4, "Dogs and cats are good friends")
    joins the postings list for `cat`.
    """
    stemmed = build_index(lab4_docs, Analyzer())
    assert lab4_index.postings("cat") == [0, 2, 5]
    assert stemmed.boolean_and("cat") == [0, 2, 3, 5]


def test_boolean_and_analyses_the_query(toy_index):
    """A user types `CATS`; the index holds `cat`. Both go through one analyser."""
    assert toy_index.boolean_and("CATS") == toy_index.boolean_and("cat")
    assert toy_index.boolean_and("cat dog") == [0, 2]


def test_boolean_and_with_an_unknown_term_is_empty(toy_index):
    assert toy_index.boolean_and("cat zzzzzz") == []


def test_a_query_that_analyses_to_nothing_returns_nothing(toy_index):
    """`the the` is all stop words, so the analyser returns no terms at all.

    An AND over an empty term list has no rarest list to start from; return an
    empty result rather than raising on `min()` of an empty sequence.
    """
    assert toy_index.analyse_query("the the") == []
    assert toy_index.boolean_and("the the") == []
    assert toy_index.boolean_and("") == []


def test_boolean_and_not(toy_index):
    assert toy_index.boolean_and_not("cat", "dog") == [5]


def test_boolean_or(toy_index):
    assert toy_index.boolean_or("bird mat") == [0, 3, 4, 5]


def test_rarest_first_does_not_change_the_answer(toy_index):
    assert toy_index.boolean_and("cat dog") == toy_index.boolean_and("dog cat")


# ── the incidence matrix ─────────────────────────────────────────────────

def test_matrix_agrees_with_the_index(toy_docs, analyzer, toy_index):
    matrix = build_matrix(toy_docs, analyzer)
    assert matrix.M.dtype == bool, "a cell records presence, not a count"
    assert matrix.M.shape == (len(matrix.vocab), len(toy_docs))
    assert matrix.vocab == sorted(matrix.vocab)
    assert matrix.query(["cat", "dog"]) == toy_index.boolean_and("cat dog")
    assert matrix.query(["cat"], ["dog"]) == toy_index.boolean_and_not("cat", "dog")


def test_matrix_row_of_an_unknown_term_is_all_false(toy_docs, analyzer):
    """`query` is the only caller of `row`, so it is easy to leave `row` unwritten
    and inline the lookup. Later work trips on that, so it is checked here."""
    matrix = build_matrix(toy_docs, analyzer)
    row = matrix.row("zzzzzz")
    assert row.shape == (len(toy_docs),)
    assert not row.any()


def test_matrix_stats_are_consistent(toy_docs, analyzer):
    stats = build_matrix(toy_docs, analyzer).stats()
    assert stats["cells"] == stats["terms"] * stats["docs"]
    assert 0 < stats["nonzero"] <= stats["cells"]


def test_projected_size_grows_with_the_product():
    """The claim your report has to make is about how the cost grows."""
    small = projected_size(20_000, 10_000)
    big = projected_size(20_000, 1_000_000)
    assert big["cells"] == 100 * small["cells"]


@pytest.mark.slow
def test_the_full_corpus(reuters_index):
    assert reuters_index.n_docs == 10_788
    # A sanity bound, not a specification: every one of the ablation settings
    # falls inside it, so this catches a broken build rather than a wrong
    # pipeline. The idf values and P@10 in test_ranking.py are what pin the
    # pipeline itself.
    assert 15_000 < reuters_index.vocabulary_size() < 30_000, \
        f"{reuters_index.vocabulary_size():,} terms — something is badly wrong with the build"
    assert reuters_index.boolean_and("crude oil opec") == \
        sorted(reuters_index.boolean_and("opec oil crude"))
    assert len(reuters_index.boolean_and("crude oil opec")) == 72
