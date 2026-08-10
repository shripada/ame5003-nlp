# Classroom demos

The scripts that were run live in class. Each one prints its way through a
single idea, using the same examples as the lesson it belongs to, so what you
run here matches what you read.

These are for following along and experimenting. They are not assessed, and
nothing in the labs depends on them.

## Setup

You need [uv](https://docs.astral.sh/uv/getting-started/installation/). Then,
once, after cloning:

```
cd classroom
uv sync                # creates .venv and installs everything
uv run bootstrap.py    # downloads the spaCy model and NLTK corpora
```

`uv sync` installs the packages. `bootstrap.py` fetches the two things pip does
not: spaCy's English model and NLTK's corpora. Skip it and a demo will stop with
`OSError: Can't find model 'en_core_web_sm'`.

Both commands are safe to run again.

## Running a demo

```
uv run demos/s04_unicode.py
```

There is nothing to activate. `uv run` uses `.venv` on its own.

| file | session | shows |
| --- | --- | --- |
| `demos/s01_regex_intro.py` | 01 | regular expressions from scratch, and why `re` is not enough |
| `demos/s02_regex_greedy.py` | 02 | groups, non-capturing groups, and the greedy trap |
| `demos/s03_bpe.py` | 03 | byte pair encoding, trained step by step on a small corpus |
| `demos/s04_unicode.py` | 04 | Unicode: why two identical-looking strings are not equal |

## Notes

- Python is pinned to 3.12 in `.python-version`. spaCy and gensim publish
  wheels for 3.12; on a newer Python they often have none yet and fall back to
  building from source, which is slow and can fail. uv downloads 3.12 for you,
  so this does not depend on the Python already on your machine.
- Each demo stands alone and can be read top to bottom. There is no shared
  helper module to look up.
- `.venv/` is not in the repository. `pyproject.toml`, `uv.lock` and
  `.python-version` are, and those three are what make the environment
  reproducible.
