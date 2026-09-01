#!/usr/bin/env python3
"""Session 11 — n-grams by sliding a window, and how little of the table a corpus fills.

    uv run demos/s11_ngrams.py

The sentence is lesson 11's, so the n-gram lists printed here are the lists in
lessons/0011-language-models-and-n-grams.html. Sections 4 and 5 put the
lesson's 50,000^n table next to a real corpus — NLTK's 18 Gutenberg books —
because the table is the size of the space the model must cover, not the number
of sequences anything has ever observed, and the gap between the two is the
whole reason n stays small.

No probability is computed here. Counting is session 12.

The short version: a window of n words, slid one step at a time. More context
narrows the guess, and empties the shelf it is estimated from.
"""

import collections

from nltk.corpus import gutenberg


def head(title: str) -> None:
    print(f"\n{title}\n{'-' * len(title)}")


def ngrams(words: list[str], n: int) -> list[tuple[str, ...]]:
    """Every run of n adjacent words, one per starting position."""
    return [tuple(words[i:i + n]) for i in range(len(words) - n + 1)]


SENTENCE = "I want to eat Indian food".split()


# ── the window ───────────────────────────────────────────────────────────
head("1. Sliding a window of width n")

print(f"    {' '.join(SENTENCE)}   ({len(SENTENCE)} words)")
print()
print(f"    {'n':>2}  {'name':9} {'how many':>8}  the n-grams")
for n, name in [(1, "unigram"), (2, "bigram"), (3, "trigram"), (4, "4-gram")]:
    grams = ngrams(SENTENCE, n)
    listed = " · ".join(" ".join(g) for g in grams)
    print(f"    {n:>2}  {name:9} {len(grams):>8}  {listed}")

T = len(SENTENCE)
print(f"""
  Each row has T - n + 1 entries, and here T = {T}: {T} unigrams, {T - 1} bigrams,
  {T - 2} trigrams, {T - 3} 4-grams. One n-gram per starting position, and the last
  n - 1 positions have no room left for a full window.""")


# ── overlap ──────────────────────────────────────────────────────────────
head("2. The windows overlap, which is the point")

for word in ["eat", "I"]:
    holding = [g for g in ngrams(SENTENCE, 3) if word in g]
    print(f"    trigrams containing '{word}':  "
          + " · ".join(" ".join(g) for g in holding))

print("""
  'eat' sits in three different trigrams, so the corpus shows it to us in
  three different contexts. 'I' sits in one, because it is at the edge.
  Every word except those at the ends is seen more than once, and that
  overlap is how a model learns what a word keeps company with.""")


# ── the model's context ──────────────────────────────────────────────────
head("3. An n-gram is a run of words; an n-gram model uses n - 1 of them")

print("    predicting the last word of 'I want to eat Indian ____'")
print()
# The history is everything before the blank; a model with window n conditions
# on the last n - 1 words of it.
history = SENTENCE[:-1]
print(f"    {'model':9} {'context it conditions on':24} question it asks")
for n, name in [(1, "unigram"), (2, "bigram"), (3, "trigram"), (4, "4-gram")]:
    context = history[len(history) - (n - 1):] if n > 1 else []
    shown = " ".join(context) if context else "nothing at all"
    question = (f"what usually follows '{shown}'?" if context
                else "how common is this word?")
    print(f"    {name:9} {shown:24} {question}")

print("""
  The off-by-one is worth saying out loud: a trigram model is a model over
  trigrams, so it conditions on two words, not three. The n counts the words
  in the window, and one of them is the word being predicted.""")


# ── the table, and the corpus ────────────────────────────────────────────
head("4. How big the table is, and how much of it a corpus fills")

# 18 books, each token lowercased, alphabetic tokens only — the same treatment
# as session 4 and session 10, so the vocabulary is comparable with theirs.
words = [w.lower() for w in gutenberg.words() if w.isalpha()]
V = len(set(words))

print(f"    {len(gutenberg.fileids())} books, {len(words):,} tokens, "
      f"vocabulary V = {V:,} distinct words")
print()
print(f"    {'model':9} {'cells to fill (V^n)':>27} "
      f"{'observed':>11}   how full")
for n, name in [(1, "unigram"), (2, "bigram"), (3, "trigram"), (4, "4-gram")]:
    possible = V ** n
    observed = len(set(ngrams(words, n)))
    print(f"    {name:9} {possible:>27,} {observed:>11,}   "
          f"1 cell in {possible / observed:>17,.0f}")

print("""
  The middle column is the table the model must be able to answer from: one
  cell for every sequence it could be handed, including the ones nobody has
  ever written down. The right-hand column is what two million words of
  English actually put in it.

  A corpus of T tokens supplies at most T n-grams however large n is, one per
  position, so the observed column barely moves while the number of cells is
  multiplied by V at every step. Bigrams reach one cell in three thousand;
  trigrams, one in fifty million. That is what 'the counts get sparser' means,
  and it is not a small-corpus problem: more text adds to a column that grows
  by addition, against one that grows by multiplication.""")

bigrams = collections.Counter(ngrams(words, 2))
once = sum(1 for c in bigrams.values() if c == 1)
print(f"""
  Worse, most of what was observed was observed once. Of the {len(bigrams):,}
  distinct bigrams here, {once:,} occur exactly one time — {once / len(bigrams):.0%} of
  them. A cell holding a single observation is filled, but it is not really
  estimated from.""")


# ── why we still want the context ────────────────────────────────────────
head("5. What the longer context buys, in the same corpus")

trigrams = collections.Counter(ngrams(words, 3))

after_one: collections.Counter = collections.Counter()
for (first, second), count in bigrams.items():
    if first == "their":
        after_one[second] += count

after_two: collections.Counter = collections.Counter()
for (first, second, third), count in trigrams.items():
    if (first, second) == ("opened", "their"):
        after_two[third] += count

print(f"    seen after 'their'          {len(after_one):>5,} different continuations, "
      f"{sum(after_one.values()):>6,} occurrences")
print("      the commonest few:        "
      + ", ".join(w for w, _ in after_one.most_common(6)))
print()
print(f"    seen after 'opened their'   {len(after_two):>5,} different continuations, "
      f"{sum(after_two.values()):>6,} occurrences")
print("      all of them:              "
      + ", ".join(w for w, _ in after_two.most_common()))

print(f"""
  This is lesson 11's exercise, run on real text. A bigram model asked to
  continue 'the students opened their' sees the one word 'their', and
  {len(after_one):,} different words follow 'their' in these books: a set so wide
  that the answer is barely constrained. One more word of context cuts the
  candidates to {len(after_two)}, every one of them a thing that can be opened.

  Both halves of the tension are on the screen at once. The longer context is
  the informative one, and it is informative because it is rare, which is
  exactly why there is so little evidence behind it: {sum(after_two.values())} occurrences
  against {sum(after_one.values()):,}. Section 4 is the same fact counted over the whole
  vocabulary.""")


# ── forward ──────────────────────────────────────────────────────────────
head("6. What is still missing")

print("    P(food | Indian) = ?")
print("""
  Nothing here is a probability yet. We have windows, contexts and counts of
  how many things were seen, but no number saying how likely 'food' is after
  'Indian'. Turning the counts into probabilities is session 12, and the
  zeros in section 4 — the cells nothing filled — are session 13.""")

print()
