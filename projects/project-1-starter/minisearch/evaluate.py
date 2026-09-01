"""Stage 5 — measuring your own choices.

The harness is given. What you measure with it is the largest block of marks in
the project.

Precision at 10 here is a rough measure and you should treat it as one. A
document about the oil price that Reuters happened not to label `crude` counts
against you unfairly, and ten queries is a small sample. It is nevertheless a
great deal better than looking at the results and deciding they seem reasonable,
and it has the property that matters: it is computed the same way before and
after a change, so a difference means something.
"""

from minisearch import corpus
from minisearch.pipeline import Analyzer

#: The ten supplied queries, each with the Reuters topic label a good answer
#: ought to carry. Do not change these — your numbers are comparable with
#: everyone else's only because the query set is fixed.
QUERIES = [
    ("crude oil prices opec", "crude"),
    ("interest rate cut by the federal reserve", "interest"),
    ("wheat grain exports to the soviet union", "grain"),
    ("coffee quota talks", "coffee"),
    ("gold mining output", "gold"),
    ("japanese yen dollar currency intervention", "money-fx"),
    ("company fourth quarter earnings per share", "earn"),
    ("merger acquisition takeover bid", "acq"),
    ("sugar production quota", "sugar"),
    ("shipping freight rates", "ship"),
]

#: The rows of the ablation. The first four add one step at a time. The fifth is
#: the full pipeline with the stop list taken back out, and it is there to test
#: session 10's claim that idf suppresses common words without needing one:
#: compare it against "+ Porter stem", which is the same pipeline with the list.
SETTINGS = [
    ("fold only", Analyzer(fold=True, drop_stop=False, stem=False)),
    ("+ stop words", Analyzer(fold=True, drop_stop=True, stem=False)),
    ("+ Porter stem", Analyzer(fold=True, drop_stop=True, stem=True)),
    ("+ lemmatize", Analyzer(fold=True, drop_stop=True, stem=False, lemma=True)),
    ("stem, no stop list", Analyzer(fold=True, drop_stop=False, stem=True)),
]


def precision_at_k(search, k=10, labels=corpus.labels):
    """Mean fraction of the top k carrying the query's paired label. Given.

    `search` is a function taking (query, k) and returning [(doc_id, score), ...].
    `labels` is a function taking a document ID and returning its topic labels;
    it is an argument only so that the tests can check this harness against a
    collection whose right answers are known, without loading Reuters.
    """
    total = 0.0
    for query, label in QUERIES:
        hits = 0
        for doc_id, score in search(query, k):
            if label in labels(doc_id):
                hits += 1
        total += hits / k
    return total / len(QUERIES)


def per_query_precision(search, k=10, labels=corpus.labels):
    """The same, query by query, so you can see which ones your system fails.

    Returns a list of (query, label, precision) triples.
    """
    rows = []
    for query, label in QUERIES:
        hits = 0
        for doc_id, score in search(query, k):
            if label in labels(doc_id):
                hits += 1
        rows.append((query, label, hits / k))
    return rows


def ablation():
    """One row per entry in SETTINGS.

    Each row is a dictionary: the setting's name, the vocabulary size, how long
    the index took to build, the mean query time in milliseconds, and the mean
    precision at 10.

    Read the documents once with `corpus.all_raw()` and reuse that list for
    every setting, so that the only thing changing between rows is the analyser.
    That is the whole reason the analyser is an argument rather than a set of
    module-level flags.

    Return the rows; printing them is `cli.py`'s job.
    """
    raise NotImplementedError("stage 5")
