"""Stage 1. These test the contract, not your particular expressions.

Your regexes will differ from anyone else's and that is fine. What is not fine
is a headline that keeps its newline, or a money finder that returns strings
with no digit in them.
"""

import pytest

from minisearch import corpus
from minisearch.extract import find_dates, find_money, split_document

SAMPLE = (
    "ASIAN EXPORTERS FEAR DAMAGE FROM U.S.-JAPAN RIFT\n"
    "  Mounting trade friction between the\n"
    "  U.S. And Japan has raised fears among many of Asia's exporting\n"
    "  nations that the row could inflict far-reaching economic damage.\n"
)


def test_headline_is_the_first_line():
    assert split_document(SAMPLE)["headline"] == "ASIAN EXPORTERS FEAR DAMAGE FROM U.S.-JAPAN RIFT"


def test_body_is_unwrapped():
    body = split_document(SAMPLE)["body"]
    assert "\n" not in body, "the hard wrapping should be collapsed to single spaces"
    assert body.startswith("Mounting trade friction between the U.S.")
    assert "  " not in body, "collapse runs of whitespace, do not leave double spaces"


def test_a_document_with_no_body_still_has_both_keys():
    doc = split_document("JUST A HEADLINE\n")
    assert set(doc) >= {"headline", "body"}
    assert doc["headline"] == "JUST A HEADLINE"
    assert doc["body"] == ""


@pytest.mark.parametrize("text", [
    "The company earned 3.5 mln dlrs in the quarter.",
    "Shipments totalled 1,250,000 tonnes.",
    "Prices rose 2.1 pct on the day.",
])
def test_money_is_found_in_each_common_shape(text):
    found = find_money(text)
    assert found, f"nothing found in {text!r}"
    assert all(any(ch.isdigit() for ch in m) for m in found), \
        "a money match with no digit in it is not a money match"


def test_money_finds_nothing_where_there_is_nothing():
    assert find_money("The chairman resigned without comment.") == []


@pytest.mark.parametrize("text", ["dated March 5, 1987", "in MARCH 1987", "on 5 MARCH 1987"])
def test_dates_are_found(text):
    assert find_dates(text), f"nothing found in {text!r}"


@pytest.mark.slow
def test_money_is_common_across_the_corpus():
    """Most Reuters documents quote a number. If yours does not find them, the
    expression is too narrow to be worth reporting on."""
    hits = 0
    for text in corpus.all_raw(500):
        if find_money(text):
            hits += 1
    assert hits > 300, f"money found in only {hits} of the first 500 documents"


@pytest.mark.slow
def test_every_document_splits_without_raising():
    for doc_id, text in enumerate(corpus.all_raw(2000)):
        doc = split_document(text)
        assert doc["headline"], f"document {doc_id} produced an empty headline"
