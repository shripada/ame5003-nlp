#!/usr/bin/env python3
"""Session 12 — bigram probabilities by counting, and the zero at the end of it.

    uv run demos/s12_bigram_probabilities.py

The four-sentence corpus is lesson 12's, and every number below is counted from
it rather than typed in beside it, so the table here is the table in
lessons/0012-n-gram-probabilities-and-the-markov-assumption.html. The last
section reaches the zero the lesson ends on, and stops there — the repair is
session 13.

The short version: divide the count of a pair by the count of the word it
starts with. Multiply along the sentence. One unseen pair takes the product to
zero.
"""

import collections

CORPUS = [
    "<s> I want to eat Indian food </s>",
    "<s> I want to eat Chinese food </s>",
    "<s> I want Indian food </s>",
    "<s> she wants to eat Indian food </s>",
]


def head(title: str) -> None:
    print(f"\n{title}\n{'-' * len(title)}")


# Counts, taken once and used everywhere below. Bigrams are counted inside a
# sentence only: </s> is followed by nothing, so no pair ever spans two
# sentences.
unigram: collections.Counter = collections.Counter()
bigram: collections.Counter = collections.Counter()
for sentence in CORPUS:
    words = sentence.split()
    unigram.update(words)
    bigram.update(zip(words, words[1:]))


def p_bigram(first: str, second: str) -> float | None:
    """The MLE P(second | first) = count(pair) / count(first), or None if unseen.

    None is not the same as 0.0 and the difference matters in section 7: a pair
    that never occurred has probability 0, while a pair whose first word never
    occurred has no estimate at all, because the division is by zero.
    """
    if unigram[first] == 0:
        return None
    return bigram[(first, second)] / unigram[first]


# ── the corpus ───────────────────────────────────────────────────────────
head("1. The whole training corpus")

for sentence in CORPUS:
    print(f"    {sentence}")

vocabulary = sorted(set(unigram) - {"<s>"})
print(f"""
    {sum(unigram.values())} tokens in all, {len(vocabulary)} of them predictable

  <s> gives the first word of a sentence something to condition on, so we can
  ask how likely 'I' is at the start rather than treating it as unconditioned.
  </s> is how the model stops: it is an ordinary token the model can predict,
  and predicting it ends the sentence. <s> is never predicted, only ever
  conditioned on, which is why it is not in the {len(vocabulary)}:""")
print(f"\n    {', '.join(vocabulary)}")


# ── the chain rule ───────────────────────────────────────────────────────
head("2. The chain rule is exact, and unusable")

history = ("I", "want", "to", "eat", "Indian")
full = history + ("food",)
n_full = sum(1 for s in CORPUS if " ".join(full) in s)
n_history = sum(1 for s in CORPUS if " ".join(history) in s)

print(f"    P(food | {' '.join(history)})")
print(f"      count({' '.join(full)})    = {n_full}")
print(f"      count({' '.join(history)})         = {n_history}")
print(f"      estimate                            = {n_full}/{n_history} "
      f"= {n_full / n_history:.1f}")

unseen = ("we", "want", "to", "eat", "Indian")
n_unseen = sum(1 for s in CORPUS if " ".join(unseen) in s)
n_unseen_full = sum(1 for s in CORPUS if " ".join(unseen + ("food",)) in s)
print(f"""
    P(food | {' '.join(unseen)})
      count({' '.join(unseen + ("food",))})   = {n_unseen_full}
      count({' '.join(unseen)})        = {n_unseen}
      estimate                            = {n_unseen_full}/{n_unseen} — undefined

  The first estimate is 1.0 on a denominator of {n_history}, which is one observation
  and no basis for a probability. The second divides by {n_unseen}, and the sentence
  it asks about is perfectly ordinary English. Long histories almost never
  recur, so the chain rule's terms are either uncountable or counted once.""")


# ── the Markov assumption ────────────────────────────────────────────────
head("3. The Markov assumption: condition on one word instead")

print(f"""    P(food | {' '.join(history)})   ->   P(food | Indian)

  This is false as a claim about language and we adopt it anyway. 'I' at the
  start of the sentence really does bear on the last word, and we discard it,
  because a long history is correct and uncountable while a short one is
  wrong and countable. count(Indian) = {unigram['Indian']} in four sentences; the five-word
  history above occurred {n_history}.""")


# ── the counts ───────────────────────────────────────────────────────────
head("4. Every bigram probability, counted")

print(f"    {'bigram':18} {'count(pair)':>11} {'count(first)':>12} {'estimate':>16}")
for (first, second), count in sorted(bigram.items(), key=lambda kv: -kv[1]):
    estimate = p_bigram(first, second)
    print(f"    {first + ' ' + second:18} {count:>11} {unigram[first]:>12} "
          f"{f'{count}/{unigram[first]} = {estimate:.3f}':>16}")

print("""
  count(pair) / count(first word), and nothing else. Of the three times 'eat'
  occurs, two are followed by 'Indian', so the estimate is 2/3. The denominator
  is always the count of the history word on its own — never the corpus size,
  which is the inversion to watch for.""")


# ── a distribution ───────────────────────────────────────────────────────
head("5. Each context gives a distribution over next words")

for context in ["eat", "want", "<s>", "Indian"]:
    continuations = {second: count for (first, second), count in bigram.items()
                     if first == context}
    parts = ", ".join(f"P({w} | {context}) = {c}/{unigram[context]}"
                      for w, c in sorted(continuations.items(), key=lambda kv: -kv[1]))
    total = sum(continuations.values()) / unigram[context]
    print(f"    {parts}    sums to {total:.1f}")

print(f"""
  Every row sums to 1, because the model is a distribution over what comes
  next, not a score. Note the last one: P(food | Indian) = 1.0 says only that
  these four sentences never put another word after 'Indian'. In a real corpus
  'Indian' is followed by hundreds of words and the largest of them might be a
  few per cent. Three rows here are doing honest work — <s> to I at
  {p_bigram('<s>', 'I'):.2f}, want to to at {p_bigram('want', 'to'):.2f}, eat to Indian at
  {p_bigram('eat', 'Indian'):.2f} — and the rest are artefacts of a tiny corpus.""")


# ── scoring ──────────────────────────────────────────────────────────────
head("6. Scoring a sentence: multiply along it")

def score(sentence: str, show: bool = False) -> float:
    """The bigram probability of a whole sentence, one factor per adjacent pair.

    Stops at the first factor it cannot compute, which happens only when a word
    is missing from the vocabulary altogether.
    """
    words = sentence.split()
    total = 1.0
    for first, second in zip(words, words[1:]):
        estimate = p_bigram(first, second)
        if estimate is None:
            if show:
                print(f"      P({second:8} | {first:8}) = "
                      f"{bigram[(first, second)]}/{unigram[first]} — undefined, "
                      f"'{first}' is not in the vocabulary")
            break
        if show:
            print(f"      P({second:8} | {first:8}) = "
                  f"{bigram[(first, second)]}/{unigram[first]} = {estimate:.3f}")
        total *= estimate
    return total


print("    <s> I want to eat Indian food </s>")
detail = score(CORPUS[0], show=True)
print(f"      product                = {detail:.3f}")

print("\n    and the same for the others:\n")
for sentence in CORPUS:
    print(f"      {score(sentence):.3f}   {sentence}")

print("""
  Which is the model doing its job: it prefers 'Indian' to 'Chinese' after
  'eat', and a sentence starting 'I' to one starting 'she', and it ranks them
  on counts alone with no idea what any of the words mean.

  The third line is lesson 12's exercise. It is shorter than the first and
  still scores lower, because 'want' is followed by 'Indian' in only 1 of 3
  cases against 'to' in 2 of 3. That the shorter sentence loses is an accident
  of a corpus full of probabilities equal to 1.0; in real data every extra
  word multiplies in another factor well below 1, so longer sentences score
  lower almost automatically. Comparing sentences of different lengths by raw
  probability is not sound, and session 14 is about what to do instead.""")


# ── the zero ─────────────────────────────────────────────────────────────
head("7. Where it fails")

FAILING = "<s> I want to eat Indian pizza </s>"
print(f"    {FAILING}\n")
score(FAILING, show=True)
print(f"      product                = {score(FAILING):.3f}")

print(f"""
  Not a small probability. Exactly zero, because count(Indian pizza) = 0 makes
  one factor 0/{unigram['Indian']}, and one zero anywhere in a product takes the whole thing
  with it. Every other factor was fine.

  The last line of the trace is a second and different failure, worth naming
  now and repairing later: 'pizza' is not in the vocabulary at all, so
  P(</s> | pizza) divides by count(pizza) = 0 and there is no estimate to
  multiply in, zero or otherwise.

  The model is asserting that an ordinary English sentence is impossible, on
  the evidence that four sentences of training text did not happen to contain
  it. Real corpora make this worse rather than better: most bigrams that could
  occur never do, so most test sentences contain at least one unseen pair and
  score exactly zero. A model that calls most of English impossible is not a
  model anyone can use.""")

print()
