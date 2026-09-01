# Project 1 — a mini search engine

Starter code for project 1 of AME 5053. Read
[the brief](../project-1-mini-search-engine.html) first: it says what the system
has to do and how it is marked. This file says how to run what is here.

If you would rather build the whole thing from an empty directory, the
[requirements sheet](../project-1-requirements.md) is the same project written
out in plain English with no code. Both routes are marked identically.

## Setup

You need **Python 3.10 or later** — the code uses `list[str]` and `int | None`
annotations, which are syntax errors on 3.9.

Work from inside this directory. `pytest` and `python -m minisearch.cli` both
resolve `minisearch` relative to where you are, so from the course-repo root
neither finds it.

```
cd projects/project-1-starter
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python -c "from minisearch import corpus; corpus.ensure_corpora()"
```

The last line downloads the Reuters corpus and NLTK's stop words and lemmatizer
data, about 10 MB. It is safe to run again.

## What is here

```
minisearch/
  corpus.py     the collection, and the document ordering everyone uses    given
  extract.py    headline, body, amounts, dates                             stage 1
  pipeline.py   raw text -> index terms                                    stage 2
  matrix.py     the term-document incidence matrix                         stage 3
  index.py      the inverted index and its postings merges                 stage 3
  ranking.py    TF-IDF weights and ranked retrieval                        stage 4
  evaluate.py   the precision@10 harness (given) and the ablation          stage 5
  cli.py        a command line over all of it                              given
build_index.py  builds and saves the indexes the app loads                 given
app.py          the Streamlit interface                                    stage 6
tests/          the suite that tells you whether any of it is right
```

Every stub raises `NotImplementedError` with the stage it belongs to, so a full
test run at the start tells you the order to work in.

## Running the tests

```
pytest                                    the fast tests, about a second
pytest -m slow                            only the corpus tests, a few minutes
pytest -m ""                              both
pytest --ignore=tests/test_yours.py       only the tests you were given
pytest tests/test_pipeline.py             one stage
pytest -k merge -x                        one thing, stopping at the first failure
pytest --pdb                              drop into the debugger where it broke
```

The first run is noisy: the five `test_yours.py` tests fail on purpose and the
unwritten stubs raise `NotImplementedError` through the fixtures. Use
`--ignore=tests/test_yours.py` while you are working through the stages, and one
file at a time to see the order within a stage.

The fast tests run on a six-document collection, where you already know the
right answers, so a failure names the merge that is wrong rather than telling you
the corpus is disappointing. The slow ones build the real index and check the
numbers the brief quotes; run them before you submit and before you believe any
figure in your report.

`tests/test_yours.py` fails on purpose. Those are the tests you write, and they
carry marks.

## Debugging without the app

`cli.py` exists so you can try a query without starting Streamlit and without
adding a print statement to a module:

```
python -m minisearch.cli --limit 500 stats
python -m minisearch.cli extract 4174 8210          stage 1, on two documents
python -m minisearch.cli matrix                     stage 3's sizes and the projection
python -m minisearch.cli search "crude oil prices opec"
python -m minisearch.cli boolean "crude oil opec"
python -m minisearch.cli term opec said the
python -m minisearch.cli explain "gold mining output" --doc 4174
python -m minisearch.cli evaluate
python -m minisearch.cli ablate
```

`--no-stem`, `--keep-stop` and `--lemma` work on any of them, so you can inspect
any row of the ablation the same way you inspect the default.

`--limit 500` indexes only the first 500 documents, which turns a minute into a
second while you are still getting things wrong. `explain` prints the per-term
contributions to one document's score; when a result surprises you, run that
first, because the term carrying the score is usually not the one you expected.

## Running the app

```
python build_index.py --all
streamlit run app.py
```

`build_index.py` writes `data/indexes.pkl`, which is not committed — it is
regenerated, and it is large.

## Submitting

The brief has the full list. In short: this repository with the stubs filled in,
`report.md`, a `README.md` that takes a reader from a clone to a search box, and
a note of where you used an AI assistant. Every number in the report must come
out of code in this repository. Between them, `cli ablate` (the four settings and
the fifth stop-word run), `cli matrix` (the matrix sizes and the projection),
`cli term` (any document frequency or idf) and `pytest -m ""` reproduce all
of them. The money-count and the rarest-first timings are the two you will need
to write a few lines for yourself.
