"""Stage 4 — ranked retrieval with TF-IDF (session 10).

Boolean retrieval returns a set. On this corpus `crude AND oil AND opec` returns
72 documents in no particular order, and nobody reads 72 documents.

    w(t, d)     = 1 + log10(tf)                 the count, compressed
    idf(t)      = log10(N / df)                 how rare the term is
    score(q, d) = sum over t in q of w(t,d) * idf(t)

Both logarithms are base 10, as in the lesson. Using base e changes every number
in your report and will not match the values the tests check.
"""

import math


def tf_weight(count):
    """`1 + log10(count)`, the compressed term frequency.

    Defined only for count > 0. A term that does not occur has weight 0, and
    that case never reaches here because it is not on the postings list.
    """
    raise NotImplementedError("stage 4")


def idf(index, term):
    """`log10(N / df)` for an already-analysed term.

    A term in every document has df = N, so its idf is exactly 0 and it
    contributes nothing to any score. A term in no document should return 0 too,
    rather than dividing by zero.
    """
    raise NotImplementedError("stage 4")


def search_ranked(index, query, k=10):
    """The top `k` documents for `query`, as [(doc_id, score), ...], best first.

    Score only the documents that can score above zero — those on the postings
    list of at least one query term. Walking all 10,788 documents for every
    query gives the same ranking by the wrong algorithm, and the timing test
    exists to catch it.

    A dictionary from document ID to a running total is the straightforward way:
    for each query term, walk its postings list and add that term's contribution
    to each document's total.

    Sort by score, highest first, and break ties by document ID so that the
    ranking is the same every time it is run.

    This returns the top `k` only. Stage 6 also has to display how many
    documents matched at all — the 328 in "10 results of 328" — and that number
    is gone by the time the slice happens, so it is `n_matching` below rather
    than something to recover from this list.
    """
    raise NotImplementedError("stage 4")


def n_matching(index, query):
    """How many documents score above zero for `query`.

    The M in stage 6's "10 results of 328 in 24 ms". It is the size of the union
    of the query terms' postings lists, and it needs no scoring at all — which is
    worth noticing, because it means the count is cheap even when k is small.
    """
    raise NotImplementedError("stage 4")


def explain(index, query, doc_id):
    """Why a document scored what it did. Given to you — it is a debugging tool.

    Returns [(term, count, idf, contribution), ...] for every query term present
    in the document, largest contribution first. When a result surprises you,
    this is the first thing to run; the term carrying the score is usually not
    the one you expected.
    """
    rows = []
    seen = set()
    for term in index.analyse_query(query):
        if term in seen:
            continue
        seen.add(term)
        counts = dict(index.counts(term))
        if doc_id not in counts:
            continue
        count = counts[doc_id]
        term_idf = idf(index, term)
        rows.append((term, count, term_idf, tf_weight(count) * term_idf))
    rows.sort(key=lambda row: -row[3])
    return rows
