#!/usr/bin/env python3
"""Session 03 — byte pair encoding, trained step by step on a small corpus.

    uv run demos/s03_bpe.py

Uses the corpus from lessons/0003-tokenization-and-the-vocabulary-problem.html
(which is the SLP3 §2.4 worked example), so the merges printed here are the
merges on the board.

BPE in one sentence: start from single characters, then repeatedly glue
together whichever adjacent pair is most common. Frequent words end up whole;
rare words stay in pieces. Nothing is ever <UNK>.
"""

import collections


def head(title: str) -> None:
    print(f"\n{title}\n{'-' * len(title)}")


# The corpus, as counts of words. '_' marks the end of a word, so the model
# can tell "er" inside 'wider' from "er" at the end of 'newer'.
CORPUS = {
    "l o w _": 5,
    "l o w e s t _": 2,
    "n e w e r _": 6,
    "w i d e r _": 3,
    "n e w _": 2,
}


def pair_counts(corpus: dict[str, int]) -> collections.Counter:
    """Count every adjacent symbol pair, weighted by how often the word occurs."""
    counts: collections.Counter = collections.Counter()
    for word, freq in corpus.items():
        symbols = word.split()
        for a, b in zip(symbols, symbols[1:]):
            counts[a, b] += freq
    return counts


def merge(corpus: dict[str, int], pair: tuple[str, str]) -> dict[str, int]:
    """Rewrite the corpus with every occurrence of `pair` glued into one symbol.

    Walk the symbol list rather than string-replacing "a b" in the joined word.
    A plain replace would also match across a symbol boundary: merging (e, s)
    on the symbols ["ne", "s", "t"] would produce ["nes", "t"], even though no
    symbol "e" is followed by "s" there.
    """
    a, b = pair
    merged: collections.Counter = collections.Counter()
    for word, freq in corpus.items():
        symbols = word.split()
        out = []
        i = 0
        while i < len(symbols):
            if i < len(symbols) - 1 and symbols[i] == a and symbols[i + 1] == b:
                out.append(a + b)
                i += 2
            else:
                out.append(symbols[i])
                i += 1
        # Counter, not dict: if two words collapse to the same symbols, their
        # counts should add rather than one overwriting the other.
        merged[" ".join(out)] += freq
    return dict(merged)


def show(corpus: dict[str, int]) -> None:
    for word, freq in corpus.items():
        print(f"    {freq:>2}  {word}")


def train(corpus: dict[str, int], num_merges: int) -> list[tuple[str, str]]:
    """Run BPE, printing the corpus after each merge. Returns the merge list."""
    vocabulary = sorted({sym for w in corpus for sym in w.split()})
    print("  starting vocabulary (single characters):")
    print(f"    {', '.join(vocabulary)}\n")
    print("  starting corpus:")
    show(corpus)

    merges = []
    for step in range(1, num_merges + 1):
        counts = pair_counts(corpus)
        if not counts:
            break

        best, freq = counts.most_common(1)[0]
        merges.append(best)
        corpus = merge(corpus, best)

        print(f"\n  merge {step}:  {best[0]} + {best[1]}  ->  {best[0]}{best[1]}"
              f"   (seen {freq} times)")
        show(corpus)

    return merges


def apply_merges(word: str, merges: list[tuple[str, str]]) -> list[str]:
    """Tokenize an unseen word by replaying the learned merges in order."""
    symbols = list(word) + ["_"]
    for a, b in merges:
        i = 0
        while i < len(symbols) - 1:
            if symbols[i] == a and symbols[i + 1] == b:
                symbols[i : i + 2] = [a + b]
            else:
                i += 1
    return symbols


head("Training BPE on the five-word corpus")
learned = train(CORPUS, num_merges=8)

head("What it learned")
print("  merges, in order:")
for i, (a, b) in enumerate(learned, 1):
    print(f"    {i}. {a} + {b}  ->  {a}{b}")

head("Now tokenize words it has never seen")
print("""\
  This is the payoff. No word is ever <UNK> — the worst case is that
  it comes back as single characters.
""")

for word in ["lower", "newest", "widest", "doomscrolling"]:
    pieces = apply_merges(word, learned)
    print(f"    {word:16} ->  {' | '.join(pieces)}")

print("""
  'lower' splits into low + er, two pieces it already knows.
  'doomscrolling' was never in the corpus, so it falls back to
  characters — still a valid tokenization, still no <UNK>.""")

print()
