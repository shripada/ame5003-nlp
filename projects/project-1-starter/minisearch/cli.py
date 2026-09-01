"""A command line over the engine. Given to you.

This exists so that you can try a query without opening Streamlit and without
adding a print statement to a module. Every subcommand works as soon as the
stubs it needs are filled in, so it is also a rough progress report.

    python -m minisearch.cli stats
    python -m minisearch.cli matrix
    python -m minisearch.cli extract 4174 8210
    python -m minisearch.cli search "crude oil prices opec"
    python -m minisearch.cli boolean "crude oil opec"
    python -m minisearch.cli explain "gold mining output" --doc 4174
    python -m minisearch.cli term opec
    python -m minisearch.cli evaluate
    python -m minisearch.cli ablate

Add `--no-stem` or `--keep-stop` to any of them to search a differently
analysed index — which is the quickest way to see stage 5's point before you
write the table.

`python -m pdb -m minisearch.cli search "..."` drops you into the debugger, and
`pytest --pdb` does the same on the first failing test.
"""

import argparse
import sys
import time

from minisearch import corpus, evaluate, ranking
from minisearch.extract import find_dates, find_money, split_document
from minisearch.index import build_index
from minisearch.matrix import build_matrix, projected_size
from minisearch.pipeline import Analyzer


def analyzer_from(args):
    if args.lemma:
        return Analyzer(drop_stop=not args.keep_stop, stem=False, lemma=True)
    return Analyzer(drop_stop=not args.keep_stop, stem=not args.no_stem)


def build(args):
    analyzer = analyzer_from(args)
    print(f"building over {args.limit or corpus.n_docs():,} documents "
          f"with {analyzer} ...", file=sys.stderr)
    started = time.perf_counter()
    index = build_index(corpus.all_raw(args.limit), analyzer)
    print(f"  {index.vocabulary_size():,} terms, {index.total_postings():,} postings, "
          f"{time.perf_counter() - started:.1f}s", file=sys.stderr)
    return index


def show(doc_id, score=None):
    doc = split_document(corpus.raw(doc_id))
    head = f"{doc_id:>6}"
    if score is not None:
        head += f"  {score:6.3f}"
    print(f"{head}  {doc['headline'][:70]}")
    print(f"{'':>14}{','.join(corpus.labels(doc_id))}")


def cmd_stats(args):
    index = build(args)
    by_df = sorted(index.entries, key=index.df, reverse=True)[:30]
    print(f"\n{'term':<16}{'df':>8}{'idf':>8}")
    for term in by_df:
        print(f"{term:<16}{index.df(term):>8,}{ranking.idf(index, term):>8.3f}")


def cmd_search(args):
    index = build(args)
    t0 = time.perf_counter()
    results = ranking.search_ranked(index, args.query, k=args.k)
    total = ranking.n_matching(index, args.query)
    ms = (time.perf_counter() - t0) * 1000
    print(f"\n{len(results)} results of {total} in {ms:.1f} ms\n")
    for doc_id, score in results:
        show(doc_id, score)


def cmd_boolean(args):
    index = build(args)
    t0 = time.perf_counter()
    hits = index.boolean_and(args.query)
    ms = (time.perf_counter() - t0) * 1000
    print(f"\n{len(hits)} documents in {ms:.1f} ms — an unordered set\n")
    for doc_id in hits[: args.k]:
        show(doc_id)


def cmd_explain(args):
    index = build(args)
    print(f"\nwhy document {args.doc} scores what it does for {args.query!r}:\n")
    print(f"{'term':<16}{'count':>7}{'idf':>8}{'contribution':>14}")
    for term, count, term_idf, contribution in ranking.explain(index, args.query, args.doc):
        print(f"{term:<16}{count:>7}{term_idf:>8.3f}{contribution:>14.3f}")


def cmd_term(args):
    index = build(args)
    for raw_term in args.terms:
        for term in index.analyse_query(raw_term):
            print(f"{raw_term!r} -> {term!r}  df={index.df(term):,}  "
                  f"idf={ranking.idf(index, term):.3f}  "
                  f"first postings={index.postings(term)[:10]}")


def cmd_extract(args):
    """Stage 1, on one document — so "read twenty at random" is a command."""
    for doc_id in args.docs:
        raw = corpus.raw(doc_id)
        doc = split_document(raw)
        print(f"\n{doc_id}  {corpus.fileids()[doc_id]}  {corpus.labels(doc_id)}")
        print(f"  headline: {doc['headline']}")
        print(f"  body:     {doc['body'][:200]}...")
        print(f"  money:    {find_money(raw)[:8]}")
        print(f"  dates:    {find_dates(raw)[:8]}")


def cmd_matrix(args):
    """Stage 3's measurements, and the extrapolation the report argues from."""
    analyzer = analyzer_from(args)
    subset = args.limit or 500
    matrix = build_matrix(corpus.all_raw(subset), analyzer)
    stats = matrix.stats()
    print(f"\nover {stats['docs']:,} documents:")
    for key in ("terms", "cells", "nonzero"):
        print(f"  {key:<10}{stats[key]:>16,}")
    print(f"  {'density':<10}{stats['density']:>15.2%}")
    print(f"  {'as bool':<10}{stats['bytes_as_bool'] / 1e6:>15.1f} MB")
    print(f"  {'packed':<10}{stats['bytes_packed'] / 1e6:>15.1f} MB")

    index = build_index(corpus.all_raw(), analyzer)
    sizes = [
        (stats["terms"], stats["docs"]),
        (index.vocabulary_size(), corpus.n_docs()),
        (500_000, 1_000_000),
    ]
    print("\nprojected, at three sizes:")
    print(f"  {'terms':>10}{'docs':>12}{'cells':>18}{'packed MB':>14}")
    for terms, docs in sizes:
        projected = projected_size(terms, docs)
        print(f"  {terms:>10,}{docs:>12,}{projected['cells']:>18,.0f}"
              f"{projected['mb_packed']:>14,.0f}")
    print(f"\nthe inverted index over the same {corpus.n_docs():,} documents holds "
          f"{index.total_postings():,} postings")


def cmd_evaluate(args):
    index = build(args)

    def search(query, k):
        return ranking.search_ranked(index, query, k)

    print(f"\n{'query':<45}{'label':<10}{'P@10':>6}")
    for query, label, precision in evaluate.per_query_precision(search, k=args.k):
        print(f"{query:<45}{label:<10}{precision:>6.1f}")
    print(f"\nmean P@{args.k} = {evaluate.precision_at_k(search, k=args.k):.2f}")


def cmd_ablate(args):
    rows = evaluate.ablation()
    if not rows:
        print("ablation() returned nothing — stage 5 is not done yet.")
        return
    columns = list(rows[0])
    widths = [max(len(c), *(len(f"{r[c]}") for r in rows)) + 2 for c in columns]
    print("".join(c.ljust(w) for c, w in zip(columns, widths)))
    for row in rows:
        print("".join(f"{row[c]}".ljust(w) for c, w in zip(columns, widths)))


def main(argv=None):
    parser = argparse.ArgumentParser(prog="minisearch")
    parser.add_argument("--limit", type=int, default=None,
                        help="index only the first N documents (faster while debugging)")
    parser.add_argument("--no-stem", action="store_true")
    parser.add_argument("--keep-stop", action="store_true")
    parser.add_argument("--lemma", action="store_true",
                        help="lemmatize instead of stemming (the fourth ablation row)")
    parser.add_argument("-k", type=int, default=10)
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("stats").set_defaults(func=cmd_stats)
    sub.add_parser("matrix").set_defaults(func=cmd_matrix)
    sub.add_parser("ablate").set_defaults(func=cmd_ablate)
    sub.add_parser("evaluate").set_defaults(func=cmd_evaluate)

    for name, func in (("search", cmd_search), ("boolean", cmd_boolean)):
        p = sub.add_parser(name)
        p.add_argument("query")
        p.set_defaults(func=func)

    p = sub.add_parser("explain")
    p.add_argument("query")
    p.add_argument("--doc", type=int, required=True)
    p.set_defaults(func=cmd_explain)

    p = sub.add_parser("term")
    p.add_argument("terms", nargs="+")
    p.set_defaults(func=cmd_term)

    p = sub.add_parser("extract")
    p.add_argument("docs", nargs="+", type=int)
    p.set_defaults(func=cmd_extract)

    args = parser.parse_args(argv)
    args.func(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
