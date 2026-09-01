"""The collection. Given to you — do not change the ordering.

Everyone indexes the same documents in the same order, so that a document ID
means the same thing in your submission as in everyone else's. Document `i` is
`fileids()[i]`, and `fileids()` is `sorted(reuters.fileids())`. A claim about
document 4174 can then be checked.
"""

import nltk

NLTK_PACKAGES = ["reuters", "stopwords", "punkt", "punkt_tab", "wordnet", "omw-1.4"]

# Loading the corpus takes a moment, so we do it once and keep it in these two
# module-level variables. They start as None and are filled in the first time
# anything asks for them.
_reuters = None
_fileids = None


def ensure_corpora():
    """Download what NLTK needs. Safe to call repeatedly."""
    for package in NLTK_PACKAGES:
        nltk.download(package, quiet=True)


def _load():
    """The NLTK reuters object, loaded on first use."""
    global _reuters
    if _reuters is None:
        ensure_corpora()
        from nltk.corpus import reuters
        _reuters = reuters
    return _reuters


def fileids():
    """The 10,788 document IDs, in the one order this project uses."""
    global _fileids
    if _fileids is None:
        _fileids = sorted(_load().fileids())
    return _fileids


def n_docs():
    """How many documents there are."""
    return len(fileids())


def raw(doc_id):
    """The raw text of document `doc_id`."""
    return _load().raw(fileids()[doc_id])


def labels(doc_id):
    """The Reuters topic labels of document `doc_id`.

    These were assigned by Reuters, not by us. Your search engine must never
    read them: they exist in this project only so that stage 5 can check whether
    a change improved the results or made them worse.
    """
    return list(_load().categories(fileids()[doc_id]))


def all_raw(limit=None):
    """The raw text of every document, as a list, in document-ID order.

    Pass `limit` to read only the first N — much faster while you are still
    getting things wrong.
    """
    if limit is None:
        limit = n_docs()
    return [raw(doc_id) for doc_id in range(limit)]
