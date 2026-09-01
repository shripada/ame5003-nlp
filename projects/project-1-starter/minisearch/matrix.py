"""Stage 3, first half — the term-document incidence matrix (session 8).

Built over a 500-document subset, because the point of this module is to measure
the thing rather than to use it. `projected_size` is where that measurement
turns into an argument about scale.
"""

import numpy as np


class IncidenceMatrix:
    """`M[t, d]` is True when term `vocab[t]` occurs in document `d`.

    Build one with `build_matrix` below.
    """

    def __init__(self, vocab, M, analyzer):
        self.vocab = vocab          # every distinct term, sorted
        self.M = M                  # a numpy array of bool, terms x documents
        self.analyzer = analyzer

    def row(self, term):
        """The row for `term` — an all-False row if it is not in the vocabulary.

        `self.vocab` is a sorted list, so `list.index` works but scans. That is
        fine here: this matrix exists to be measured, not to be fast.
        """
        raise NotImplementedError("stage 3")

    def query(self, and_terms, not_terms=()):
        """`a AND b AND NOT c`, answered with bitwise operations on the rows.

        The terms arrive raw, as a user typed them, so run them through
        `self.analyzer` first — the same rule as everywhere else.

        Return a list of document positions. `np.flatnonzero` turns a row of
        True and False into the positions of the Trues.
        """
        raise NotImplementedError("stage 3")

    def stats(self):
        """Cells, non-zero cells, and the two sizes, for your report. Given."""
        cells = int(self.M.size)
        nonzero = int(self.M.sum())
        return {
            "terms": len(self.vocab),
            "docs": int(self.M.shape[1]),
            "cells": cells,
            "nonzero": nonzero,
            "density": nonzero / cells,
            "bytes_as_bool": int(self.M.nbytes),
            "bytes_packed": cells // 8,
        }


def build_matrix(raw_docs, analyzer):
    """Build the matrix over a list of raw document strings.

    `vocab` is every distinct term across the whole collection, sorted. The
    array's dtype should be `bool`: a cell records presence, not a count.
    """
    raise NotImplementedError("stage 3")


def projected_size(n_terms, n_docs):
    """What the matrix would cost at a given size. Given to you.

    Use it on the full corpus and then on session 8's web-sized numbers. Read
    both results before you write that paragraph of your report: at Reuters
    scale the answer is not the one session 8's argument might lead you to
    expect, and the honest claim is about how the cost grows rather than what it
    happens to be here.
    """
    cells = n_terms * n_docs
    return {"cells": cells, "mb_as_bool": cells / 1e6, "mb_packed": cells / 8 / 1e6}
