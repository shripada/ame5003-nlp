"""Stage 6 — the interface.

    python build_index.py --all      once
    streamlit run app.py

A skeleton, not a specification: the brief lists what the page has to show and
this gets you as far as a search box. The parts marked TODO are yours.
"""

import pathlib
import pickle
import time

import streamlit as st

from minisearch import ranking

INDEX_FILE = pathlib.Path("data/indexes.pkl")


@st.cache_resource
def load():
    """Load once per process, not once per keystroke.

    Without the decorator Streamlit re-runs this file top to bottom on every
    interaction, rebuilding nothing but re-reading everything, and the app is
    unusable.
    """
    if not INDEX_FILE.exists():
        st.error(f"{INDEX_FILE} is missing — run `python build_index.py --all` first.")
        st.stop()
    try:
        with INDEX_FILE.open("rb") as fh:
            return pickle.load(fh)
    except Exception as exc:                      # a half-written or stale pickle
        st.error(f"could not read {INDEX_FILE} ({exc}). "
                 "Delete it and run `python build_index.py --all` again.")
        st.stop()


def snippet(body, index, terms, width=300):
    """A window of the body with the matched words marked.

    The highlight has to survive stemming: a search for `mining` should mark
    `mines` if that is why the document matched, which means you cannot look for
    the user's string in the text. Analyse each word of the body and compare the
    terms — and say in your report how you did it.
    """
    raise NotImplementedError("stage 6")


# Streamlit requires set_page_config to be the first Streamlit command in the
# script, and load() calls st.error on a missing index file — so the page config
# has to come first, before anything touches the data.
st.set_page_config(page_title="Reuters search", layout="wide")

data = load()
indexes, docs = data["indexes"], data["docs"]

st.title("Reuters search")

with st.sidebar:
    st.header("Pipeline")
    st.caption(
        "Each setting has its own index behind it. An analyser cannot be swapped "
        "after the fact — the terms in the index were produced by it — so the "
        "switch changes which index is searched, not how the query is analysed."
    )
    setting = st.radio("Index", list(indexes), index=0)

index = indexes[setting]
st.caption(f"{index.vocabulary_size():,} terms · {index.total_postings():,} postings · "
           f"{index.analyzer}")

query = st.text_input("Search", "crude oil prices opec")
mode = st.radio("Retrieval", ["Ranked (TF-IDF)", "Boolean (AND)"], horizontal=True)

if query:
    started = time.perf_counter()
    # TODO: run the query in the selected mode and show, for each result:
    #   the headline, the score, the topic labels, and a snippet with the
    #   query's terms highlighted; then a line reading
    #   "10 results of 328 in 24 ms".
    #
    #   The 328 is ranking.n_matching(index, query) in ranked mode and
    #   len(hits) in Boolean mode — search_ranked returns the slice, not the
    #   total.
    st.warning("Stage 6 is not finished yet.")
