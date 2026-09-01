"""Stage 3, second half — the inverted index and its postings merges (session 9).

Merges over sorted lists, not `set()` intersections. A set intersection gives
the same answer and is genuinely fast, but it hides the merge, and the merge is
what session 9 was about. Use sets anywhere else you like.
"""


def intersect(p1, p2):
    """AND, as a linear merge over two sorted lists of document IDs.

    Walk both lists with one pointer each. Never `set(p1) & set(p2)`, and never
    `in`, which is a linear scan inside a loop and makes this quadratic.

    Do not sort the inputs defensively either: a postings list is sorted by
    construction, and sorting here would cost more than the merge saves. One
    test checks this by passing unsorted lists and requiring the wrong answer.
    """
    raise NotImplementedError("stage 3")


def union(p1, p2):
    """OR, as a linear merge. The result stays sorted and has no duplicates."""
    raise NotImplementedError("stage 3")


def difference(p1, p2):
    """AND NOT — everything in p1 that is not in p2, as a linear merge."""
    raise NotImplementedError("stage 3")


class InvertedIndex:
    """A dictionary from term to a sorted list of (document ID, count) pairs.

    The count is stored now because stage 4 needs it, and adding it later means
    rebuilding. The analyser is stored because a query has to go through the
    same one the documents did.

    Build one with `build_index` below rather than filling this in by hand.
    """

    def __init__(self, analyzer):
        self.analyzer = analyzer
        self.n_docs = 0
        self.entries = {}          # term -> [(doc_id, count), ...]

    # ── reading the index ────────────────────────────────────────────────

    def counts(self, term):
        """The postings list of an already-analysed term, with counts."""
        return self.entries.get(term, [])

    def postings(self, term):
        """The document IDs for an already-analysed term."""
        return [doc_id for doc_id, count in self.counts(term)]

    def df(self, term):
        """Document frequency — the length of the postings list, no extra pass."""
        return len(self.counts(term))

    def vocabulary_size(self):
        return len(self.entries)

    def total_postings(self):
        total = 0
        for postings in self.entries.values():
            total += len(postings)
        return total

    # ── querying ─────────────────────────────────────────────────────────

    def analyse_query(self, query):
        """The query, through the index's own analyser. Given to you.

        This one line is the whole defence against the mismatch that this
        project is built to catch, which is why it lives on the index instead of
        being something you have to remember at every call site.
        """
        return self.analyzer.terms(query)

    def boolean_and(self, query):
        """AND over every term of `query`, rarest postings list first.

        Session 9's optimisation: process terms in increasing document
        frequency, so the intermediate result is as small as possible as early
        as possible. A term that is not in the index makes the whole conjunction
        empty, and a query with no terms at all returns an empty list.
        """
        raise NotImplementedError("stage 3")

    def boolean_or(self, query):
        """Documents matching at least one term of `query`."""
        raise NotImplementedError("stage 3")

    def boolean_and_not(self, query, excluded):
        """Documents matching every term of `query` and none of `excluded`."""
        raise NotImplementedError("stage 3")


def build_index(raw_docs, analyzer):
    """Index a list of raw document strings and return an InvertedIndex.

    A document's ID is its position in `raw_docs`. For each document, run the
    analyser over it and count how often each term appears —
    `collections.Counter` does that in one line — then append (doc_id, count) to
    that term's postings list.

    Feed the documents in order and every postings list comes out sorted by
    document ID, which is what the merges above depend on.
    """
    raise NotImplementedError("stage 3")
