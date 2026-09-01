"""Stage 1 — pulling structure out of raw text, with regular expressions.

A Reuters document is a headline in capitals on its own line, then a body that
is hard-wrapped mid-sentence. That is a piece of structure sitting inside
unstructured text, and sessions 1 and 2 are how you get it out.

Use the `regex` module, not `re` (lab 2 explains why).
"""

import regex


def split_document(raw: str) -> dict[str, str]:
    """Raw Reuters text -> {"headline": ..., "body": ...}.

    The headline is the first line, stripped. The body is everything after it
    with the hard wrapping collapsed to single spaces, which is what makes a
    readable snippet later.

    A document with no body should still return both keys.
    """
    raise NotImplementedError("stage 1")


def find_money(text: str) -> list[str]:
    """Every money or quantity expression in `text`, as it appears.

    The corpus writes them in several shapes: "3.5 mln dlrs", "1,250,000",
    "2.1 pct", "vs 500 tonnes". Catch the common forms. You will not catch them
    all and you are not expected to — what you are expected to do is look at
    what you missed and say so in your report.
    """
    raise NotImplementedError("stage 1")


def find_dates(text: str) -> list[str]:
    """Every date-like expression in `text`.

    The corpus writes them as "March 5", "MARCH 1987", "5 MARCH 1987" and
    other variants.
    """
    raise NotImplementedError("stage 1")
