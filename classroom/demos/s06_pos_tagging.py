#!/usr/bin/env python3
"""Session 06 — part-of-speech tagging, including one sentence spaCy gets wrong.

    uv run demos/s06_pos_tagging.py

Same sentences as lessons/0006-part-of-speech-tagging.html, tagged with spaCy
(en_core_web_sm). The garden-path sentence at the end really is mis-tagged;
that is the point of showing it.

The short version: a word does not have a fixed grammatical class. Only the
sentence around it decides, which is why tagging is a sequence-labelling task
and not a dictionary lookup.
"""

import spacy
from nltk.corpus import gutenberg, stopwords


def head(title: str) -> None:
    print(f"\n{title}\n{'-' * len(title)}")


# en_core_web_sm is spaCy's small English pipeline — tokenizer, POS tagger and
# lemmatizer trained together. pip does not install it with spaCy, so
# `uv run bootstrap.py` downloads it; without that this line raises
# OSError: Can't find model 'en_core_web_sm'. Calling nlp(text) runs the
# pipeline and returns a Doc we can iterate over as tokens.
nlp = spacy.load("en_core_web_sm")


def tag_row(sentence: str) -> None:
    """Print a sentence with its Universal tags lined up underneath."""
    # is_punct is set by the tokenizer, so dropping punctuation costs nothing
    # extra. The tags are the point here, and a column for '.' is noise.
    doc = [t for t in nlp(sentence) if not t.is_punct]
    # One column per token, wide enough for whichever of the two rows is longer,
    # so the tag always sits directly under the word it belongs to.
    width = [max(len(t.text), len(t.pos_)) + 2 for t in doc]
    print("    " + "".join(t.text.ljust(w) for t, w in zip(doc, width)))
    print("    " + "".join(t.pos_.ljust(w) for t, w in zip(doc, width)))


# ── what the tagger produces ─────────────────────────────────────────────
head("1. Every word gets a label")

tag_row("They book flights daily.")
print("""
  That is the whole output of a POS tagger: one grammatical class per
  word. What makes it a real task is the next section.""")


# ── the same word, three jobs ────────────────────────────────────────────
head("2. The same four letters, twice a verb and once a noun")

# Tag each sentence in full, then pull out the one token we want to compare.
# The tagger has to see the whole sentence — that is the claim being made — so
# there is no way to ask it about 'book' on its own.
for sentence in ["Give me the book.", "I can book a flight.", "They book flights daily."]:
    doc = nlp(sentence)
    book = [t for t in doc if t.text.lower() == "book"][0]
    print(f"    {sentence:28}  book -> {book.pos_}")

print("""
  No dictionary can choose between these; it would only list both. The
  neighbours decide. After 'the' comes a noun, and after a modal such as
  'can' comes a verb. The tagger reads the whole sentence, which is why
  this is called a sequence-labelling task.""")


# ── the two tagsets ──────────────────────────────────────────────────────
head("3. Two tagsets, same sentence")

print(f"    {'word':10} {'Universal':12} {'Penn Treebank':10}")
print(f"    {'-' * 10} {'-' * 12} {'-' * 13}")
for token in nlp("They book flights daily."):
    if token.is_punct:
        continue
    # Every token carries both tagsets at once: .pos_ is the coarse Universal
    # tag, .tag_ the fine-grained Penn Treebank one. spacy.explain turns a tag
    # into the English sentence it stands for, which saves looking up the table.
    print(f"    {token.text:10} {token.pos_:12} {token.tag_:6} ({spacy.explain(token.tag_)})")

print("""
  Universal has about 17 tags and is readable across languages. Penn has
  about 45 and splits each class further — plural vs singular noun, which
  present tense. Neither is more correct. Take the coarse one when you
  only need noun-vs-verb, as the lemmatizer in session 5 did.""")


# ── the callback to session 5 ────────────────────────────────────────────
head("4. This is the information lemmatization was missing")

for sentence in ["We had a meeting.", "They are meeting now."]:
    doc = nlp(sentence)
    meeting = [t for t in doc if t.text == "meeting"][0]
    print(f"    {sentence:24}  {meeting.pos_:5} ->  lemma  {meeting.lemma_}")

print("""
  Session 5 left this open: the lemmatizer needed the part of speech and
  we deferred it to 'a tool that works it out'. This is that tool.""")


# ── open and closed classes ──────────────────────────────────────────────
head("5. Open classes and closed classes")

print("""\
  Open classes take new members — 'to google', 'selfie' — and are the
  content words: NOUN, VERB, ADJ, ADV.
  Closed classes almost never do: DET, PRON, ADP, AUX, CCONJ, SCONJ,
  PART.
  Nobody is coining a new word for 'the'.""")

# The closed classes and a stop-word list are close to the same set of words.
# That is not a coincidence, and it is worth measuring rather than asserting.
CLOSED = {"DET", "PRON", "ADP", "AUX", "CCONJ", "SCONJ", "PART"}
STOP = set(stopwords.words("english"))

# 20,000 words of running English, joined back into a string because spaCy
# tokenizes for itself. That is enough for the two percentages to settle, and
# it tags in a few seconds rather than a few minutes.
sample = " ".join(w for w in gutenberg.words("austen-emma.txt")[:20000] if w.isalpha())
tokens = [t for t in nlp(sample) if t.is_alpha]
closed = [t for t in tokens if t.pos_ in CLOSED]
open_class = [t for t in tokens if t.pos_ in {"NOUN", "VERB", "ADJ", "ADV"}]

# Now ask the same question of both groups: what share of these words does a
# frequency-built stop-word list happen to contain? lower() because NLTK's list
# is lower case throughout.
on_list = sum(1 for t in closed if t.text.lower() in STOP)
open_on_list = sum(1 for t in open_class if t.text.lower() in STOP)

print()
print(f"  In {len(tokens):,} words of Austen's Emma:")
print(f"    closed-class words on NLTK's stop-word list   "
      f"{on_list:,}/{len(closed):,}  ({100 * on_list / len(closed):.0f}%)")
print(f"    open-class words on the same list             "
      f"{open_on_list:,}/{len(open_class):,}  ({100 * open_on_list / len(open_class):.0f}%)")
print("""
  The stop-word list of session 4 is very nearly the closed classes,
  arrived at from the other direction — by frequency rather than by
  grammar. Same words, two different reasons.""")


# ── the honest failure ───────────────────────────────────────────────────
head("6. A sentence the tagger gets wrong")

print("""\
  'The old man the boat.' is grammatical English. It means 'the old
  people staff the boat' — 'the old' is standing in as a noun, and 'man'
  is the verb, as in to man a lifeboat.""")
print()
tag_row("The old man the boat.")
print("""
  spaCy tags 'man' as a NOUN and 'old' as an ADJ, which is the reading
  you took on your first pass too, and it is a dead end. These are called
  garden-path sentences. Taggers run at about 97% accuracy, and the last
  few percent look like this. It is not a bug to be fixed by a better
  dictionary — the ambiguity is in the language.""")

print()
