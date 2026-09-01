"""Stage 2 — raw text to index terms.

One analyser, every step switchable. The analyser is used on documents at index
time and on queries at search time, and the reason it has to be the same object
both times is the single most common way this project goes wrong:

    If the index holds `opec` because the analyser folded case, and a user's
    `OPEC` reaches the index unfolded, the term is simply absent and the query
    silently returns nothing.

Session 9 said this. The inverted index keeps the analyser it was built with and
uses it on every query, so the two cannot drift apart.
"""

import unicodedata

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.tokenize import word_tokenize

from minisearch import corpus

# Built once on first use, like the corpus, because loading them is slow.
_stop_words = None
_stemmer = PorterStemmer()
_lemmatizer = WordNetLemmatizer()


def english_stopwords():
    """NLTK's English stop word list, as a set."""
    global _stop_words
    if _stop_words is None:
        corpus.ensure_corpora()
        _stop_words = set(stopwords.words("english"))
    return _stop_words


class Analyzer:
    """Text in, index terms out.

    fold       NFKC-normalise and case fold             (session 4)
    drop_stop  remove NLTK's English stop words         (session 4)
    stem       Porter stemming                          (session 5)
    lemma      WordNet lemmatization, instead of stem   (session 5)

    Make one and hand it to `build_index`. The index keeps it and uses it on
    every query, which is what stops the query and the documents being
    normalised differently.
    """

    def __init__(self, fold=True, drop_stop=True, stem=True, lemma=False):
        if stem and lemma:
            raise ValueError("choose stemming or lemmatization, not both")
        self.fold = fold
        self.drop_stop = drop_stop
        self.stem = stem
        self.lemma = lemma

    def terms(self, text):
        """Apply every enabled step, in this order. The order is part of the
        specification, not a suggestion:

        1. NFKC-normalise the whole string, always — before tokenizing, because
           a compatibility ligature has to become letters before anything can
           split on word boundaries. This does not depend on `fold`.
        2. tokenize with NLTK's `word_tokenize` (NOT str.split, NOT a regex —
           session 3 spent an hour on why both fail)
        3. keep only tokens that are all letters, with `token.isalpha()`. That
           drops punctuation and numerals; the numbers you pulled out in stage 1
           are metadata, not index terms.
        4. case fold, if `fold`
        5. discard stop words, if `drop_stop`. Compare the lowercased token
           whether or not `fold` is set — NLTK's list is lowercase, so testing an
           unfolded token would silently keep every capitalised `The`.
        6. stem, or lemmatize, or neither

        Return a list of strings, in the order they appeared. Repeats stay in:
        stage 4 counts them.
        """
        raise NotImplementedError("stage 2")

    def describe(self):
        """A short label for the settings, for printing in tables."""
        steps = []
        if self.fold:
            steps.append("fold")
        if self.drop_stop:
            steps.append("stop")
        if self.stem:
            steps.append("stem")
        if self.lemma:
            steps.append("lemma")
        return "+".join(steps) if steps else "none"

    def __repr__(self):
        return f"Analyzer({self.describe()})"


#: The pipeline this project's numbers are quoted against.
DEFAULT = Analyzer()
