#!/usr/bin/env python3
"""Session 10 — TF-IDF, and what each of the two logarithms is doing.

    uv run demos/s10_tfidf.py

The idf table is IIR's Reuters collection, the same numbers as
lessons/0010-ranking-with-tf-idf.html, computed here from the document
frequencies rather than quoted. The last two sections rank a real collection —
NLTK's 18 Gutenberg books — so the weighting can be watched working.

The short version: often here, rare everywhere. One log for diminishing
returns on the count, one for the range of word rarity.
"""

import collections
import math

from nltk.corpus import gutenberg


def head(title: str) -> None:
    print(f"\n{title}\n{'-' * len(title)}")


def tf_weight(count: int) -> float:
    """The log-weighted term frequency, 1 + log10(tf), and 0 when tf is 0.

    The weight is only defined for tf > 0, which is why log(0) is never taken:
    a term absent from a document contributes nothing rather than negative
    infinity.
    """
    return 1 + math.log10(count) if count > 0 else 0.0


def idf(n_documents: int, document_frequency: int) -> float:
    """log10(N / df) — high for a rare term, and exactly 0 for one in every document."""
    return math.log10(n_documents / document_frequency)


# ── the first log ────────────────────────────────────────────────────────
head("1. Why the count goes through a logarithm")

print("    a document containing 'car' ...        weight 1 + log10(tf)")
for count in [1, 10, 100, 1000]:
    print(f"      {count:>4} time{'s' if count > 1 else ' '}                          "
          f"{tf_weight(count):.1f}")

print("""
  A document with the word 100 times is more about cars than one with it
  once, but not 100 times more. Relevance rises with the count and rises
  less each time. Every tenfold increase adds exactly 1, so 1 to 10 counts
  for as much as 10 to 100. That is the first logarithm's whole job.""")


# ── the second log ───────────────────────────────────────────────────────
head("2. Why rarity goes through a logarithm too")

# IIR's Reuters figures. N and the four document frequencies are the book's;
# the idf column is computed from them, which is the point of printing it.
N_REUTERS = 806_791
REUTERS_DF = {"auto": 6_723, "car": 18_165, "insurance": 19_241, "best": 25_235}

print(f"    a collection of N = {N_REUTERS:,} documents")
print()
print(f"    {'term':12} {'in how many docs (df)':>22}  {'idf':>6}")
for term, df in REUTERS_DF.items():
    print(f"    {term:12} {df:>22,}  {idf(N_REUTERS, df):>6.2f}")

print("""
  The rarer the word, the higher its idf: 'auto' outranks 'best' because
  fewer documents contain it. Two documents sharing 'insurance' is a far
  stronger signal than two sharing 'the', and this is the arithmetic that
  says so.""")


# ── the limiting case ────────────────────────────────────────────────────
head("3. A word in every document scores exactly zero")

print(f"    a word in all {N_REUTERS:,} documents:")
print(f"      df = N, so idf = log10({N_REUTERS:,} / {N_REUTERS:,}) "
      f"= log10(1) = {idf(N_REUTERS, N_REUTERS):.1f}")

print("""
  Session 4 removed stop words with a hand-written list. This does the
  same work with no list at all: 'the' is in nearly every document, so its
  idf is near zero and it contributes almost nothing to any score, whatever
  its count. The weighting achieves what the list was for.""")


# ── the product ──────────────────────────────────────────────────────────
head("4. The two put together")

print("    tf-idf  =  (1 + log10 tf)  x  log10(N / df)")
print("""
    high     when the term is frequent here and rare across the collection
    low      when it is rare here
    near 0   when it occurs everywhere""")

print()
print("  The exercise, worked. Both documents match the query 'car insurance':")
print()
# Document A is about insurance; document B merely uses a common word a lot.
# The idfs are the Reuters ones above, so the two hands-on numbers and the
# table agree.
idf_insurance = idf(N_REUTERS, REUTERS_DF["insurance"])
# 'the' is not in IIR's table. Taking it to be in every document makes its idf
# exactly 0, which is the limiting case above; in a real collection it is a
# shade above 0 and the argument is unchanged.
idf_the = idf(N_REUTERS, N_REUTERS)

a = tf_weight(9) * idf_insurance
b_the = tf_weight(50) * idf_the
b_insurance = tf_weight(1) * idf_insurance

print(f"    A: 'insurance' 9 times      (1 + log10 9) x {idf_insurance:.2f} = "
      f"{tf_weight(9):.2f} x {idf_insurance:.2f} = {a:.2f}")
print(f"    B: 'the' 50 times           (1 + log10 50) x {idf_the:.2f} = "
      f"{tf_weight(50):.2f} x {idf_the:.2f} = {b_the:.2f}")
print(f"       'insurance' once         (1 + log10 1) x {idf_insurance:.2f} = "
      f"{tf_weight(1):.2f} x {idf_insurance:.2f} = {b_insurance:.2f}")
print()
print(f"    A scores {a:.2f}, B scores {b_the + b_insurance:.2f}, so A ranks first.")
print("""
  Which is the right outcome: A is genuinely about insurance, and B only
  uses a common word often. The 50 occurrences of 'the' bought B nothing.""")


# ── a real collection ────────────────────────────────────────────────────
head("5. The same weighting over a real collection")

# 18 books from NLTK's Gutenberg corpus, each treated as one document. Small
# enough to build in about a second, large enough that the idfs are not toy
# numbers. Counting only alphabetic tokens, lowercased, as in session 4.
counts = {
    fileid: collections.Counter(w.lower() for w in gutenberg.words(fileid) if w.isalpha())
    for fileid in gutenberg.fileids()
}
N = len(counts)

# Document frequency: how many of the 18 books contain each word. Counting the
# distinct words of a document, so a word occurring 500 times in one book still
# adds 1.
document_frequency: collections.Counter = collections.Counter()
for words in counts.values():
    document_frequency.update(words.keys())

print(f"    {N} books, {len(document_frequency):,} distinct words")
print()
print(f"    {'term':12} {'df':>4} {'idf':>6}   {'count in Moby Dick':>19} {'tf-idf':>8}")
moby = counts["melville-moby_dick.txt"]
for term in ["the", "and", "sea", "captain", "whale", "harpooneer"]:
    df = document_frequency[term]
    weight = tf_weight(moby[term]) * idf(N, df)
    print(f"    {term:12} {df:>4} {idf(N, df):>6.2f}   {moby[term]:>19,} {weight:>8.2f}")

ratio = moby["whale"] // moby["harpooneer"]
print(f"""
  'the' occurs in all 18 books, so its idf is exactly 0 and its weight is 0
  despite appearing {moby['the']:,} times — the limiting case of section 3,
  happening for real.

  The top of the column is worth a moment. 'whale' appears {moby['whale']:,} times
  and 'harpooneer' {moby['harpooneer']}, a ratio of about {ratio} to 1, and yet
  'harpooneer' scores higher, because it is in 2 books where 'whale' is in
  6. The log on the count flattened that 15-fold difference to well under
  double, so rarity decided it. Often here matters; rare everywhere matters
  more.""")


# ── ranking ──────────────────────────────────────────────────────────────
head("6. Ranking the collection for a query")

def score(words: collections.Counter, query: list[str]) -> float:
    """Sum the tf-idf weights of the query terms — the document's score."""
    return sum(tf_weight(words[term]) * idf(N, document_frequency[term])
               for term in query if document_frequency[term])


for query in [["whale", "sea"], ["the", "and"], ["god", "heaven"]]:
    print(f"    query: {' '.join(query)}")
    ranked = sorted(counts, key=lambda f: score(counts[f], query), reverse=True)
    for fileid in ranked[:3]:
        print(f"      {score(counts[fileid], query):5.2f}  {fileid}")
    print()

print("""\
  The first query puts Moby Dick on top and the third puts the Bible there,
  which is the answer anyone would give. The second query scores every one
  of the 18 books at exactly 0.00, so the three shown are simply the first
  three loaded: 'the' and 'and' are in all 18 books and carry no weight, so
  a query of nothing but common words ranks nothing. That is the correct
  behaviour, and no stop-word list was consulted to get it.

  Boolean search of sessions 8 and 9 returned an unordered set. This
  returns an order. In session 18 these weights become a whole vector per
  document, one number per term, and similarity becomes the angle between
  two vectors.""")

print()
