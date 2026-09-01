"""The collections the tests run on, and nothing else.

Most tests here use a six-document toy collection, so that the suite you run
every few minutes takes about a second and tells you exactly which merge is
wrong. The tests that need all 10,788 Reuters documents are marked `slow` and
are skipped by default:

    pytest              the fast tests, about a second
    pytest -m slow      only the corpus tests, a few minutes
    pytest -m ""        both

`pytest.ini` is what turns the slow ones off by default; there is no clever code
in this file, only the fixtures below.
"""

import pytest

from minisearch import corpus
from minisearch.index import build_index
from minisearch.pipeline import Analyzer

#: Lab 4's own six documents, copied exactly. Under an analyser configured the
#: way lab 4's tokenizer was — lowercase, no stop-word removal, no stemming —
#: this index reproduces every number lab 4 printed, allowing for lab 4 numbering
#: its documents from 1 and this project numbering them from 0.
LAB4_DOCS = [
    "The cat sat on the mat.",
    "The dog sat on the log.",
    "The cat and the dog played together.",
    "Dogs and cats are good friends.",
    "A bird sang in the tall tree.",
    "The cat chased the bird up the tree.",
]

#: The collection the fast tests actually run on: the same shape as lab 4's —
#: six short documents about cats, dogs and birds — but with a headline on each,
#: so `split_document` has something to do, and with `dog` repeated, so a term
#: has a count above 1 for stage 4 to weight.
TOY_DOCS = [
    "CATS AND DOGS\n  The cat sat on the mat with a dog.",
    "THE DOG\n  A dog barked at another dog.",
    "CAT AND DOG TOGETHER\n  The cat and the dog are friends.",
    "BIRDS\n  A bird sang in the tree.",
    "THE MAT\n  The mat was blue.",
    "CAT AND BIRD\n  A cat watched a bird.",
]


@pytest.fixture(scope="session")
def analyzer():
    return Analyzer()


@pytest.fixture(scope="session")
def toy_docs():
    return list(TOY_DOCS)


@pytest.fixture(scope="session")
def toy_index(toy_docs, analyzer):
    return build_index(toy_docs, analyzer)


@pytest.fixture(scope="session")
def lab4_docs():
    return list(LAB4_DOCS)


@pytest.fixture(scope="session")
def lab4_analyzer():
    """Configured the way lab 4's tokenizer was: lowercase, everything kept."""
    return Analyzer(fold=True, drop_stop=False, stem=False)


@pytest.fixture(scope="session")
def lab4_index(lab4_docs, lab4_analyzer):
    return build_index(lab4_docs, lab4_analyzer)


@pytest.fixture(scope="session")
def reuters_index(analyzer):
    """The real thing. Built once for the whole session — it is not cheap."""
    corpus.ensure_corpora()
    return build_index(corpus.all_raw(), analyzer)
