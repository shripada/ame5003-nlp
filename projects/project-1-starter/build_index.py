"""Build the indexes and save them, so the app does not have to. Given to you.

    python build_index.py            the default pipeline only, about a minute
    python build_index.py --all      one index per ablation setting

Building takes the better part of a minute per setting; the app must start in
about a second, so it loads what this writes.

One index per setting, rather than one index and a switch in the app, is not an
implementation detail. An analyser cannot be changed after the fact: the terms
in the index were produced by it, so searching a stemmed index with an unstemmed
query is the mismatch that `pipeline.py` warns about. If the app is to offer
those switches at all, each one has to have its own index behind it.
"""

import argparse
import pathlib
import pickle
import time

from minisearch import corpus, evaluate
from minisearch.extract import split_document
from minisearch.index import build_index
from minisearch.pipeline import DEFAULT

OUT = pathlib.Path("data/indexes.pkl")


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--all", action="store_true",
                        help="build one index per ablation setting, not just the default")
    args = parser.parse_args(argv)

    # DEFAULT is the same analyser as the "+ Porter stem" row of SETTINGS, so
    # --all iterates SETTINGS alone. Building both would put two identical
    # indexes behind two differently-named radio buttons in the app.
    settings = evaluate.SETTINGS if args.all else [("default", DEFAULT)]

    corpus.ensure_corpora()
    print(f"reading {corpus.n_docs():,} documents")
    raw = corpus.all_raw()

    indexes = {}
    for name, analyzer in settings:
        started = time.perf_counter()
        index = build_index(raw, analyzer)
        indexes[name] = index
        print(f"  {name:<20}{index.vocabulary_size():>8,} terms  "
              f"{index.total_postings():>10,} postings  "
              f"{time.perf_counter() - started:>5.1f}s")

    docs = []
    for doc_id, text in enumerate(raw):
        doc = split_document(text)
        doc["labels"] = corpus.labels(doc_id)
        docs.append(doc)

    OUT.parent.mkdir(exist_ok=True)
    with OUT.open("wb") as fh:
        pickle.dump({"indexes": indexes, "docs": docs}, fh)
    print(f"\nwrote {OUT} ({OUT.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
