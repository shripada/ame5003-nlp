#!/usr/bin/env python3
"""Session 08 — the term-document incidence matrix, and where it breaks.

    uv run demos/s08_incidence_matrix.py

The matrix, the plays and the query are the ones in
lessons/0008-incidence-matrix-and-boolean-search.html, which are in turn IIR's
own worked example, so this matches both the board and the book.

The short version: make each term a row of bits and a Boolean query becomes
bitwise arithmetic. It is correct, it is fast, and at real scale it is 99.8%
zeros, which is why session 9 exists.
"""

import numpy as np


def head(title: str) -> None:
    print(f"\n{title}\n{'-' * len(title)}")


PLAYS = ["Antony and Cleopatra", "Julius Caesar", "The Tempest",
         "Hamlet", "Othello", "Macbeth"]

# One row per term, one column per play, 1 where the play contains the word.
# Typed out rather than counted from the texts because these seven rows are
# IIR's table 1.1 and the lesson prints them unchanged; the point of the hour
# is what you do with the table, not how it was filled in.
MATRIX = {
    "Antony":    [1, 1, 0, 0, 0, 1],
    "Brutus":    [1, 1, 0, 1, 0, 0],
    "Caesar":    [1, 1, 0, 1, 1, 1],
    "Calpurnia": [0, 1, 0, 0, 0, 0],
    "Cleopatra": [1, 0, 0, 0, 0, 0],
    "mercy":     [1, 0, 1, 1, 1, 1],
    "worser":    [1, 0, 1, 1, 1, 0],
}

TERMS = list(MATRIX)
# numpy because we want the rows to behave as bit vectors: & | ~ apply to the
# whole row at once, which is the operation the lesson performs by hand.
A = np.array([MATRIX[t] for t in TERMS])


def row(term: str) -> np.ndarray:
    return A[TERMS.index(term)]


def as_bits(vector: np.ndarray) -> str:
    return " ".join(str(b) for b in vector)


def plays_of(vector: np.ndarray) -> list[str]:
    """The play names where the vector holds a 1."""
    return [PLAYS[i] for i, bit in enumerate(vector) if bit]


# ── the table ────────────────────────────────────────────────────────────
head("1. One row per term, one column per document")

# The plays are numbered rather than spelled out across the top, because six
# play names will not fit above six single-digit columns. The numbers are the
# document IDs, which is what session 9's postings lists are made of.
label_width = max(len(t) for t in TERMS) + 2
for number, play in enumerate(PLAYS, start=1):
    print(f"    {number}  {play}")
print()
print(" " * (label_width + 4) + " ".join(str(n) for n in range(1, len(PLAYS) + 1)))
for term in TERMS:
    print(f"    {term.ljust(label_width)}{as_bits(row(term))}")

print(f"""
  Each row is a bit vector: which plays contain that word. The Caesar row
  is {as_bits(row('Caesar'))}, so it appears in every play but the third.""")


# ── the query ────────────────────────────────────────────────────────────
head("2. The query becomes bitwise arithmetic")

print("    Brutus AND Caesar AND NOT Calpurnia")
print()

brutus, caesar, calpurnia = row("Brutus"), row("Caesar"), row("Calpurnia")
# ~ on a 0/1 int array gives -1 and -2, not 1 and 0, so NOT is written as
# 1 - x. The lesson flips the bits on the board; this is that flip.
not_calpurnia = 1 - calpurnia

print(f"    Brutus            {as_bits(brutus)}")
print(f"    Caesar            {as_bits(caesar)}")
print(f"    NOT Calpurnia     {as_bits(not_calpurnia)}     "
      f"<- Calpurnia was {as_bits(calpurnia)}, flipped")
print()

both = brutus & caesar
answer = both & not_calpurnia
print(f"    Brutus AND Caesar             ->   {as_bits(both)}")
print(f"            AND NOT Calpurnia     ->   {as_bits(answer)}")
print()
print("    matching plays:")
for play in plays_of(answer):
    print(f"      {play}")

print("""
  Three rows, two bitwise operations, and the query is answered. Bitwise
  operations are among the fastest things a computer does, which is the
  whole appeal of the representation.""")


# ── the exercise ─────────────────────────────────────────────────────────
head("3. The exercise, worked")

print("    Antony AND mercy")
print()
antony, mercy = row("Antony"), row("mercy")
print(f"    Antony   {as_bits(antony)}")
print(f"    mercy    {as_bits(mercy)}")
print(f"    AND      {as_bits(antony & mercy)}")
print()
print(f"    -> {', '.join(plays_of(antony & mercy))}")
print()
print("  The two words occur together in those two plays and nowhere else.")


# ── the scale problem ────────────────────────────────────────────────────
head("4. Why this does not scale")

# The figures are the lesson's, for a collection of realistic size. Computing
# them rather than printing them lets the class change a number and watch the
# percentage refuse to move.
documents = 1_000_000
terms = 500_000
distinct_words_per_document = 1_000

cells = terms * documents
ones = documents * distinct_words_per_document   # at most: one per word, per doc
zeros = cells - ones

print(f"    documents (columns)            {documents:>18,}")
print(f"    distinct terms (rows)          {terms:>18,}")
print(f"    cells in the matrix            {cells:>18,}")
print()
print(f"    distinct words in a document   {distinct_words_per_document:>18,}")
print(f"    1s in the whole matrix, at most{ones:>18,}")
print(f"    0s                             {zeros:>18,}")
print(f"    proportion of zeros            {100 * zeros / cells:>17.1f}%")

print("""
  Half a trillion cells, of which at most a billion say anything. Almost
  all of that memory records that a word is absent from a document, which
  is nearly always true and rarely what the query asked. The matrix will
  not fit in memory, and storing it this way would be wasteful if it did.""")


# ── the size of one row ──────────────────────────────────────────────────
head("5. The same point, for one word")

# Caesar is in five of six plays here, which is unrepresentative on purpose:
# in a large collection even a common word reaches a tiny fraction of it.
in_documents = 3_000
print(f"    Suppose 'Calpurnia' appears in {in_documents:,} of the {documents:,} documents.")
print()
print(f"    as a matrix row     {documents:>12,} bits, of which {in_documents:,} are 1")
print(f"    as a list of IDs    {in_documents:>12,} numbers, and no zeros at all")
print("""
  The row grows with the collection. The list grows with the word. That
  is the whole of the correction, and the structure built from it is the
  inverted index of session 9.""")

print()
