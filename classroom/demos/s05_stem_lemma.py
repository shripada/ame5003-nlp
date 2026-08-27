#!/usr/bin/env python3
"""Session 05 — stemming vs lemmatization, on the words from the lesson.

    uv run demos/s05_stem_lemma.py

Same words as lessons/0005-stemming-and-lemmatization.html, run through the
tools the lab uses: NLTK's Porter stemmer, NLTK's WordNet lemmatizer, and
spaCy. Every root printed here is what the tool really returns, including the
ones that are wrong.

The short version: a stem is a shared key and need not be a word; a lemma is
the dictionary form and always is one. The stemmer is fast and ignorant, the
lemmatizer is slower and needs to be told the part of speech.
"""

import time

import spacy
from nltk.corpus import gutenberg
from nltk.stem import PorterStemmer, WordNetLemmatizer


def head(title: str) -> None:
    print(f"\n{title}\n{'-' * len(title)}")


# The two NLTK tools, which is the whole contrast this session is about.
# Porter is rules and nothing else, so it is ready the moment it is
# constructed. WordNet is a lookup, so it needs its dictionary on disk —
# `uv run bootstrap.py` downloads it, along with the spaCy model further down.
porter = PorterStemmer()        # chops endings by rule, knows no words
wordnet = WordNetLemmatizer()   # looks the word up, needs its part of speech


# ── the family that collapses ────────────────────────────────────────────
head("1. Four spellings, one key")

for word in ["study", "studies", "studying", "studied"]:
    print(f"    {word:10} ->  {porter.stem(word)}")

print("""
  'studi' is not an English word, and that is not a defect. The stemmer
  has one job — map related words to the same string — and it did it.
  The key is never shown to anyone; a search for 'studying' now finds a
  page that says 'studies'.""")

print()
for word in ["argue", "argued", "argues", "arguing"]:
    print(f"    {word:10} ->  {porter.stem(word)}")


# ── where rules alone fail ───────────────────────────────────────────────
head("2. Where chopping endings fails")

problems = [
    ("organization", "over-stemming: the rule took too much and changed the meaning"),
    ("was", "a rule fired that should not have, leaving a non-word"),
    ("mice", "no ending to remove, so the stemmer cannot reach 'mouse'"),
    ("better", "no relation to 'good' is recognised"),
]
for word, why in problems:
    print(f"    {word:14} ->  {porter.stem(word):14}  {why}")

print()
print("  And two words that collapse together when they should not:")
for word in ["universal", "university"]:
    print(f"    {word:14} ->  {porter.stem(word)}")
print("""\
  A search for one now matches the other. This is the price of rules
  without meaning.""")


# ── the lemmatizer, and the part of speech it needs ──────────────────────
head("3. The lemmatizer looks the word up")

print("  Same words, through WordNet, with the part of speech supplied:")

# WordNet wants the part of speech as one letter: 'n' noun, 'v' verb,
# 'a' adjective, 'r' adverb. It cannot work this out for itself, so the caller
# has to know the answer already — which is the weakness section 4 shows.
lemmas = [("mice", "n"), ("was", "v"), ("better", "a"), ("studies", "v")]
for word, pos in lemmas:
    print(f"    {word:10} ({pos}) ->  {wordnet.lemmatize(word, pos)}")

print("""
  Every one of those is a word you can look up in a dictionary. That is
  the difference: a stem is a key, a lemma is the base form of the word.""")

head("4. Without a part of speech it guesses 'noun', and is often wrong")

for word, pos in [("was", "v"), ("better", "a"), ("studying", "v")]:
    guessed = wordnet.lemmatize(word)
    told = wordnet.lemmatize(word, pos)
    verdict = "same" if guessed == told else "WRONG"
    print(f"    {word:10}  no POS -> {guessed:10}  told it is '{pos}' -> {told:8}  {verdict}")

print("""
  'was' -> 'wa' with no POS, which is no better than the stemmer managed.
  The lemmatizer is only as accurate as the part of speech it is given.""")


# ── the same word, two roots, decided by the sentence ────────────────────
head("5. The root can depend on the sentence")

# en_core_web_sm is spaCy's small English pipeline: a tokenizer, a POS tagger
# and a lemmatizer that were trained together, about 12 MB of model files. pip
# does not install it alongside spaCy, so bootstrap.py downloads it; without
# that step this line raises OSError: Can't find model 'en_core_web_sm'.
nlp = spacy.load("en_core_web_sm")

# Calling nlp(text) runs that whole pipeline and hands back a Doc, which we
# iterate over to get tokens. Each token carries what the pipeline decided:
# .pos_ is its part of speech, .lemma_ its root. The trailing underscore is
# spaCy's convention for "give me the readable string" — token.pos without it
# is the integer the model actually stores.
for sentence in ["We had a meeting.", "They are meeting now."]:
    doc = nlp(sentence)
    meeting = [t for t in doc if t.text == "meeting"][0]
    print(f"  {sentence:24}  meeting is a {meeting.pos_:5} ->  lemma  {meeting.lemma_}")

print("""
  Identical spelling, different roots, and only the surrounding words
  decide which. spaCy works out the part of speech and lemmatizes in one
  step, which is why it is the usual choice. Working out the part of
  speech is itself a whole task — that is session 6.""")


# ── side by side ─────────────────────────────────────────────────────────
head("6. Side by side")

print(f"    {'word':16} {'stem (Porter)':16} {'lemma (spaCy)':16}")
print(f"    {'-' * 16} {'-' * 16} {'-' * 16}")
sentence = "The studies were better and the mice were meeting the organization"
# is_punct is set by the tokenizer, so skipping punctuation is free. Porter is
# given the raw text of each token; spaCy has already lemmatized it in context.
for token in nlp(sentence):
    if token.is_punct:
        continue
    print(f"    {token.text:16} {porter.stem(token.text):16} {token.lemma_:16}")

print("""
  One row is worth a second look. spaCy lemmatizes 'better' to 'well',
  not 'good', because in this sentence it tagged it as an adverb rather
  than an adjective — and 'well' is the right root for the adverb. The
  lemma follows the part of speech, and the part of speech follows the
  sentence.""")


# ── why anyone still stems ───────────────────────────────────────────────
head("7. Why anyone still stems: speed")

# gutenberg.words() reads a whole book as one long list of tokens. isalpha
# drops the punctuation, and 20,000 words is enough for the ratio below to be
# stable while still finishing in a couple of seconds in front of a class.
words = [w for w in gutenberg.words("austen-emma.txt") if w.isalpha()][:20000]
text = " ".join(words)   # spaCy takes a string and tokenizes it itself

# perf_counter rather than time(): it is monotonic and has the resolution to
# measure something this short. Both timed lines build a list of one root per
# word, so the two numbers are measuring comparable work.
start = time.perf_counter()
[porter.stem(w) for w in words]
stem_time = time.perf_counter() - start

# nlp(text) does the tagging and lemmatizing eagerly, so almost all of this
# time is spent inside that call; walking the Doc to read .lemma_ is cheap.
start = time.perf_counter()
[t.lemma_ for t in nlp(text)]
lemma_time = time.perf_counter() - start

print(f"  {len(words):,} words of Austen's Emma:")
print(f"    Porter stemmer      {stem_time:6.2f} s")
print(f"    spaCy tag+lemmatize {lemma_time:6.2f} s   ({lemma_time / stem_time:.0f}x slower)")
print("""
  spaCy is doing far more work — it tags every word before it lemmatizes.
  At web scale, over billions of words whose roots are never displayed,
  that is correctness you cannot use and would still pay for. Stem for
  search; lemmatize when the root is shown, or fed to a model.""")

print()
