"""The tests you write. Worth marks; see the brief's rubric.

Everything in the other files tests the contract you were handed. These test the
decisions you made, which nobody else can write for you. Each one below fails
until you replace the `pytest.fail` with a real test — that is deliberate, so
that a full green run means the whole suite, not just the given part.

Delete none of them. Add more.
"""

import pytest


def test_your_money_expression_on_a_case_you_found():
    """Take a real document your stage 1 extractor gets wrong, and pin the
    behaviour down. Your report describes this failure; this test proves you
    actually found it rather than assumed it."""
    pytest.fail("write this one — the report claims a failure, so pin it here")


def test_your_analyser_on_a_word_that_stemming_ruins():
    """Session 5 said stemming and lemmatization are not interchangeable. Find a
    word in this corpus where Porter produces something misleading, and assert
    what each of the two does with it."""
    pytest.fail("write this one")


def test_the_corpus_specific_stop_word_you_found():
    """NLTK's stop list was written for English in general, not for newswire.
    Assert the document frequency and idf of the word you name in your report,
    so that the number in the report and the number in the code cannot drift."""
    pytest.fail("write this one")


def test_a_query_your_system_handles_badly():
    """Every system has one. Assert the behaviour you describe in your report —
    a wrong document in the top three, an empty result for a reasonable query.
    A failing search pinned by a passing test is worth more here than pretending
    the failure is not there."""
    pytest.fail("write this one")


def test_something_you_broke_and_fixed():
    """Over three weeks you will fix at least one real bug. When you do, write
    the test that would have caught it, before you fix it. That is the habit the
    rest of this suite is trying to teach."""
    pytest.fail("write this one")
