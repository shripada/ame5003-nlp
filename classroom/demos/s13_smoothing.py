#!/usr/bin/env python3
"""Session 13 — smoothing: paying for the unseen out of the seen.

    uv run demos/s13_smoothing.py

The corpus is sessions 11 and 12's four sentences, so the add-one, backoff and
interpolation numbers here are the ones in
lessons/0013-the-zero-problem-and-smoothing.html, recomputed rather than
quoted. Section 3 is the important one: add-one is shown working and then
shown to be bad, which is the reason backoff and interpolation exist.

The short version: the total is fixed at 1, so every unit of probability given
to something unseen is taken from something seen. Add-one hands it out flat.
Backoff and interpolation hand it out on evidence.
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


def counts(sentences: list[str]) -> tuple[collections.Counter, ...]:
    """Unigram, bigram and trigram counts, taken inside sentences only."""
    unigram: collections.Counter = collections.Counter()
    bigram: collections.Counter = collections.Counter()
    trigram: collections.Counter = collections.Counter()
    for sentence in sentences:
        words = sentence.split()
        unigram.update(words)
        bigram.update(zip(words, words[1:]))
        trigram.update(zip(words, words[1:], words[2:]))
    return unigram, bigram, trigram


unigram, bigram, trigram = counts(CORPUS)

# V is the number of tokens the model can predict. <s> is only ever conditioned
# on, never predicted, so it is not one of them — hence 10 here, not 11.
VOCABULARY = sorted(set(unigram) - {"<s>"})
V = len(VOCABULARY)

# The denominator of a unigram probability: every token the model could have
# predicted, which is every token except the sentence-start markers.
N_PREDICTED = sum(unigram.values()) - unigram["<s>"]


def mle(first: str, second: str) -> float:
    return bigram[(first, second)] / unigram[first]


def add_one(first: str, second: str, vocabulary_size: int = V) -> float:
    """Laplace: one extra observation for every possible next word."""
    return (bigram[(first, second)] + 1) / (unigram[first] + vocabulary_size)


# ── the problem ──────────────────────────────────────────────────────────
head("1. Where session 12 left off")

print(f"    P(pizza | Indian) = count(Indian pizza) / count(Indian) = "
      f"{bigram[('Indian', 'pizza')]}/{unigram['Indian']} = 0")
print(f"""
    P(<s> I want to eat Indian pizza </s>) = 0

  One pair the corpus did not happen to contain, and the whole sentence is
  declared impossible. This gets worse with more data rather than better: a
  50,000-word vocabulary admits 2.5 billion bigrams and no corpus holds a
  meaningful fraction of them, so most test sentences contain at least one
  unseen pair.

  The vocabulary here is {V} predictable tokens:
    {', '.join(VOCABULARY)}

  <s> is not among them. V counts what the model can predict, and <s> is only
  ever conditioned on.""")


# ── add-one ──────────────────────────────────────────────────────────────
head("2. Add-one: pretend every possible next word was seen once more")

print(f"    P(w | eat) = (count(eat w) + 1) / (count(eat) + V) "
      f"= (count + 1) / ({unigram['eat']} + {V})")
print()
denominator = unigram["eat"] + V

print(f"    {'next word after eat':22} {'count':>5} {'MLE':>18} {'add-one':>16}")
for word in ["Indian", "Chinese", "food", "each other word"]:
    count = bigram[("eat", word)]
    shown_mle = (f"{count}/{unigram['eat']} = {mle('eat', word):.3f}" if count
                 else "0 — impossible")
    shown_add_one = f"{count + 1}/{denominator} = {(count + 1) / denominator:.3f}"
    print(f"    {word:22} {count:>5} {shown_mle:>18} {shown_add_one:>16}")

print("""
  Nothing is zero any longer, so no single unseen pair can wipe out a
  sentence, and the ordering survives: 'Indian' still beats 'Chinese', which
  still beats the words never seen after 'eat'. The V in the denominator is
  not decoration — one count was added to each of V possible next words, so V
  counts were added in total, and without it the row would no longer sum
  to 1.""")


# ── the cost ─────────────────────────────────────────────────────────────
head("3. What it cost")

print(f"    P(Indian | eat)   MLE {mle('eat', 'Indian'):.3f}   ->   "
      f"add-one {add_one('eat', 'Indian'):.3f}")
print("""
  Two thirds of the evidence actually collected has been handed to words that
  never followed 'eat' even once, on no evidence at all. At V = 10 that is
  survivable. Watch what a realistic vocabulary does to it.""")

print(f"\n    P(food | Indian), a pair observed on {unigram['Indian']} of "
      f"{unigram['Indian']} opportunities:\n")
print(f"    {'vocabulary size':24} {'add-one estimate':>28}")
for size, label in [(V, f"{V} (our toy corpus)"), (1_000, "1,000"),
                    (50_000, "50,000 (realistic)")]:
    estimate = add_one("Indian", "food", size)
    print(f"    {label:24} "
          f"{f'{bigram[("Indian", "food")] + 1}/{unigram["Indian"] + size} = {estimate:.5f}':>28}")

print("""
  A bigram seen on every single opportunity, rated at eight thousandths of one
  per cent. The 50,000 words that never followed 'Indian' have absorbed almost
  all of the probability between them.

  The defect is not the size of the constant, and adding 0.01 instead of 1
  only shrinks the error — that is add-k smoothing, and it is a real method.
  The defect is that every unseen word gets the same share. Add-one has no way
  of knowing that 'curry' is a plausible continuation of 'Indian' and 'the' is
  not, so it gives them equal weight, and there are so many implausible words
  that they drown the evidence.

  Add-one is still the right tool where the vocabulary is small: it is what
  session 16 uses for Naive Bayes. For a language model it mainly shows what
  an adequate method has to do better.""")


# ── backoff ──────────────────────────────────────────────────────────────
head("4. Backoff: when the long context is empty, ask a shorter one")

print("    P(Indian | she wants)\n")
print(f"    {'try':32} {'counts':>8}   result")
attempts = [
    ("trigram — she wants Indian", trigram[("she", "wants", "Indian")],
     bigram[("she", "wants")]),
    ("bigram — wants Indian", bigram[("wants", "Indian")], unigram["wants"]),
    ("unigram — Indian", unigram["Indian"], N_PREDICTED),
]
for label, numerator, denominator in attempts:
    estimate = numerator / denominator
    verdict = (f"{estimate:.3f} — use this" if numerator
               else "zero, back off")
    print(f"    {label:32} {f'{numerator} / {denominator}':>8}   {verdict}")

flat = 1 / (unigram["wants"] + V)
print(f"""
  The phrase was never observed, but 'Indian' is a common word in this corpus,
  and the fallback says so: {unigram['Indian']}/{N_PREDICTED} = {unigram['Indian'] / N_PREDICTED:.3f}. Add-one would have
  given it {flat:.3f} — which is exactly what add-one gives a word that never
  occurred at all, because to add-one the two cases are the same case.

  One complication, worth naming and not deriving: the shorter n-gram's
  probabilities already sum to 1 on their own, so borrowing them whole would
  push the total above 1. Real backoff scales the borrowed values down by a
  discount factor. The principle is the part that matters here.""")


# ── interpolation ────────────────────────────────────────────────────────
head("5. Interpolation: use all three every time, weighted")

LAMBDAS = (0.5, 0.3, 0.2)
print("    P(w3 | w1 w2) = L1 P(w3 | w1 w2) + L2 P(w3 | w2) + L3 P(w3),"
      "   L1 + L2 + L3 = 1\n")


def interpolate(w1: str, w2: str, w3: str, show: bool = False) -> float:
    """The three estimates, weighted and added. The unigram term is the floor."""
    estimates = [
        ("trigram", f"P({w3} | {w1} {w2})",
         trigram[(w1, w2, w3)], bigram[(w1, w2)]),
        ("bigram", f"P({w3} | {w2})", bigram[(w2, w3)], unigram[w2]),
        ("unigram", f"P({w3})", unigram[w3], N_PREDICTED),
    ]
    total = 0.0
    for weight, (name, label, numerator, denominator) in zip(LAMBDAS, estimates):
        estimate = numerator / denominator if denominator else 0.0
        total += weight * estimate
        if show:
            print(f"    {name:9} {label:24} "
                  f"{f'{numerator}/{denominator} = {estimate:.3f}':>16}   x {weight}")
    return total


print("    P(Indian | I want)\n")
combined = interpolate("I", "want", "Indian", show=True)
terms = " + ".join(f"{weight} x {estimate:.3f}" for weight, estimate in zip(
    LAMBDAS, [trigram[("I", "want", "Indian")] / bigram[("I", "want")],
              bigram[("want", "Indian")] / unigram["want"],
              unigram["Indian"] / N_PREDICTED]))
print(f"      {terms} = {combined:.3f}")

print("\n    and the case backoff was given, P(Indian | she wants):\n")
floored = interpolate("she", "wants", "Indian", show=True)
print(f"      two of the three are zero, and the result is still "
      f"{floored:.3f}, not 0")

print("""
  That is the whole point of the unigram term: it is always included and never
  zero, so the combination cannot fall to zero however empty the longer
  contexts are. The long context dominates when it is reliable and the short
  one sets a floor under it.

  In the first example the trigram and bigram happen to agree at 1/3, which is
  an accident of a corpus where almost every context occurs once or twice.

  The lambdas are not guessed, and they are not fitted on the training corpus
  either — that would just reproduce the training data. A separate held-out
  portion of text is set aside and the lambdas that score it best are chosen.
  What 'best' means is session 14.""")


# ── unknown words ────────────────────────────────────────────────────────
head("6. Unknown words are a different problem, and need <UNK>")

print("""    smoothing spreads probability over the V words we know about.
    'pizza' is not one of them, so there is nothing to spread to it.

  The standard repair is to make room for the unknown before training. Every
  word occurring once in the corpus is replaced by a single token <UNK>, which
  then has real counts like any other word. At test time, any word the model
  has not seen becomes <UNK> and inherits them.""")

rare = {word for word, count in unigram.items() if count == 1}
UNK_CORPUS = [" ".join("<UNK>" if w in rare else w for w in s.split())
              for s in CORPUS]
unigram_unk, bigram_unk, _ = counts(UNK_CORPUS)
V_UNK = len(set(unigram_unk) - {"<s>"})

print(f"\n    replaced (seen once each): {', '.join(sorted(rare))}\n")
for sentence in UNK_CORPUS:
    print(f"    {sentence}")

TEST = "<s> I want to eat Indian pizza </s>".split()
mapped = ["<UNK>" if w not in unigram_unk else w for w in TEST]
print(f"\n    test sentence, after mapping:  {' '.join(mapped)}\n")

total = 1.0
for first, second in zip(mapped, mapped[1:]):
    estimate = ((bigram_unk[(first, second)] + 1)
                / (unigram_unk[first] + V_UNK))
    total *= estimate
    print(f"      P({second:6} | {first:6}) = "
          f"({bigram_unk[(first, second)]} + 1)/({unigram_unk[first]} + {V_UNK}) "
          f"= {estimate:.3f}")
print(f"      product                   = {total:.5f}")

print(f"""
  Both repairs were needed and they do different jobs. <UNK> gave 'pizza' a
  count to work with, and add-one kept the pairs it appears in off zero. The
  sentence that session 12 called impossible now scores {total:.5f}: a small
  number, which is the honest answer, because the model is reporting that the
  sentence contains a word it does not know rather than that English does not
  contain the sentence.

  That completes the n-gram model — count, assume a short history, smooth. One
  question is left, and every choice above depends on it: bigram or trigram,
  which lambdas, which smoothing? None of it can be settled without a way to
  measure whether a model is any good, which is session 14.""")

print()
