"""Stage 5 — the harness, checked without touching the corpus.

`precision_at_k` takes its label lookup as an argument precisely so that it can
be tested against a collection whose right answers are known. The small
functions below stand in for a real search engine and a real set of labels.
"""

import pytest

from minisearch import evaluate


def top_k_documents(query, k):
    """A stand-in search engine: always returns documents 0..k-1."""
    return [(doc_id, 1.0) for doc_id in range(k)]


def no_documents(query, k):
    return []


def test_the_query_set_is_intact():
    assert len(evaluate.QUERIES) == 10
    for query, label in evaluate.QUERIES:
        assert query and label


def test_a_perfect_system_scores_one():
    every_label = [label for query, label in evaluate.QUERIES]

    def labels(doc_id):
        return every_label

    assert evaluate.precision_at_k(top_k_documents, k=10, labels=labels) == pytest.approx(1.0)


def test_a_system_that_returns_nothing_relevant_scores_zero():
    def labels(doc_id):
        return ["nothing-matches-this"]

    assert evaluate.precision_at_k(top_k_documents, k=10, labels=labels) == pytest.approx(0.0)


def test_one_query_half_right_scores_one_twentieth():
    def labels(doc_id):
        # Only even-numbered documents are about crude oil.
        if doc_id % 2 == 0:
            return ["crude"]
        return ["nothing"]

    # Only the `crude` query can score at all, and on that one exactly half the
    # ten results match — so the mean over ten queries is 0.5 / 10.
    assert "crude" in [label for query, label in evaluate.QUERIES]
    assert evaluate.precision_at_k(top_k_documents, k=10, labels=labels) == pytest.approx(0.05)


def test_per_query_precision_reports_every_query():
    def labels(doc_id):
        return []

    rows = evaluate.per_query_precision(no_documents, k=10, labels=labels)
    assert len(rows) == len(evaluate.QUERIES)
    for query, label, precision in rows:
        assert precision == 0.0


def settings_fields(analyzer):
    return (analyzer.fold, analyzer.drop_stop, analyzer.stem, analyzer.lemma)


def test_the_settings_differ_from_one_another():
    """Rows that are secretly the same analyser are an ablation of nothing."""
    fields = [settings_fields(analyzer) for name, analyzer in evaluate.SETTINGS]
    assert len(set(fields)) == len(fields)


def test_the_stop_word_setting_is_comparable_with_a_stemming_one():
    """Session 10's claim needs two runs differing in exactly one field.

    `stem, no stop list` and `+ Porter stem` must be identical apart from
    `drop_stop`, or the comparison measures two things at once and says nothing.
    """
    settings = dict(evaluate.SETTINGS)
    with_list = settings["+ Porter stem"]
    without = settings["stem, no stop list"]
    assert without.drop_stop is False
    assert with_list.drop_stop is True
    assert (without.fold, without.stem, without.lemma) == \
           (with_list.fold, with_list.stem, with_list.lemma)
