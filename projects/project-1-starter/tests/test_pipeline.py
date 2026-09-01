"""Stage 2. The last test in this file is the one the project is built around."""

import pytest

from minisearch.pipeline import Analyzer

FULL = Analyzer()                                            # fold, stop, stem
PLAIN = Analyzer(fold=True, drop_stop=False, stem=False)


def test_tokenizing_splits_punctuation_off():
    assert PLAIN.terms("the cat, really!") == ["the", "cat", "really"]


def test_case_folding():
    assert PLAIN.terms("OPEC Cut") == ["opec", "cut"]
    assert Analyzer(fold=False, drop_stop=False, stem=False).terms("OPEC Cut") == ["OPEC", "Cut"]


def test_numerals_are_not_index_terms():
    assert PLAIN.terms("3.5 mln dlrs in 1987") == ["mln", "dlrs", "in"]


def test_nfkc_runs_before_tokenizing():
    """A compatibility ligature has to become letters before word_tokenize sees
    it, or `ﬁle` stays one exotic character and never matches `file`.

    This is the step order the docstring specifies, and it is the one thing in
    that order that the Reuters figures cannot check — 1987 newswire is ASCII.
    """
    assert PLAIN.terms("\ufb01le") == ["file"]
    assert PLAIN.terms("classi\ufb01ed") == ["classified"]


def test_stop_words_are_removed():
    assert "the" not in Analyzer(stem=False).terms("the oil price")
    assert "the" in PLAIN.terms("the oil price")


def test_stemming_collapses_a_family():
    assert FULL.terms("OPEC cut production quotas") == FULL.terms("opec cutting production quota")


def test_lemmatizing_is_available_and_differs_from_stemming():
    lemma = Analyzer(stem=False, lemma=True)
    assert lemma.terms("studies") != Analyzer().terms("studies"), \
        "WordNet gives a real word where Porter gives a stem"


def test_stemming_and_lemmatizing_together_is_rejected():
    with pytest.raises(ValueError):
        Analyzer(stem=True, lemma=True)


def test_an_index_keeps_the_analyser_it_was_built_with():
    """This is what stops a query being normalised differently from the index."""
    from minisearch.index import build_index

    analyzer = Analyzer()
    index = build_index(["the cat sat on the mat"], analyzer)
    assert index.analyzer is analyzer


def test_the_query_reaches_the_documents_terms():
    """The mismatch this project exists to catch.

    If a document is analysed one way and a query another, the term is absent
    from the index and the query silently returns nothing. One analyser, both
    sides, and the failure becomes impossible rather than unlikely.
    """
    document_terms = set(FULL.terms("JAPANESE EXPORTERS FEAR DAMAGE\n  Japan's exports fell."))
    for typed in ["JAPAN", "japan", "Japanese", "exporter"]:
        assert set(FULL.terms(typed)) & document_terms, f"a user typing {typed!r} would get nothing"
