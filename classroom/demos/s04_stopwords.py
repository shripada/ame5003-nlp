#!/usr/bin/env python3
"""Session 04 — stop words: the routine step that reverses a review.

    uv run demos/s04_stopwords.py

Same sentences as lessons/0004-normalization-and-stop-words.html. The Unicode
half of that lesson is demos/s04_unicode.py; this is the other half.

The short version: removing stop words is the right call for search and the
wrong call for sentiment, and the reason is that 'not' is on the list.
"""

from nltk.corpus import gutenberg, stopwords


def head(title: str) -> None:
    print(f"\n{title}\n{'-' * len(title)}")


# stopwords.words("english") returns the list as NLTK ships it — one of
# several dozen languages in the same package, downloaded by bootstrap.py. We
# hold it as a set because the only question we ever ask of it is "is this word
# on the list?", once per token, over hundreds of thousands of tokens below.
STOP = set(stopwords.words("english"))


def strip_stop_words(sentence: str, keep: set[str] = frozenset()) -> str:
    """Drop every token on the list, except the ones we explicitly keep."""
    # STOP - keep is set difference: the list minus the words we are sparing,
    # which is how section 5 rescues the negations without editing STOP itself.
    # Splitting on whitespace is enough here because these sentences carry no
    # punctuation; a real pipeline would tokenize first. The lower() is needed
    # because NLTK's list is entirely lower case, so 'The' would otherwise
    # survive while 'the' did not.
    kept = [w for w in sentence.split() if w.lower() not in STOP - keep]
    return " ".join(kept)


# ── the list itself ──────────────────────────────────────────────────────
head("1. What is actually on the list")

print(f"  NLTK's English stop-word list has {len(STOP)} words.")
print()
print("  the first 40, alphabetically:")
# A set has no order, so sort it before printing, then walk it eight at a time
# to get five readable rows rather than one very long line.
alphabetical = sorted(STOP)
for i in range(0, 40, 8):
    print(f"    {'  '.join(w.ljust(8) for w in alphabetical[i:i + 8])}")
print("    ...")


# ── the case for removing them ───────────────────────────────────────────
head("2. Why anyone removes them")

review = "this is one of the best phones I have ever bought"
print(f"  before   {review}")
print(f"  after    {strip_stop_words(review)}")
print()
print("  Eleven tokens become five, and the five carry the substance. For search")
print("  and topic-finding that is a real improvement: less to store, and the")
print("  informative words stand out.")
print()
print("  Note that 'one' survives. It is a content-free word by any reasonable")
print("  reading, and NLTK's list simply does not have it. A stop-word list is")
print("  somebody's list, not a law — check what is on the one you are using.")

# How much of a real corpus is stop words. Gutenberg is a shelf of full books,
# so this is a fair sample of running English rather than a hand-picked line.
words = [w.lower() for w in gutenberg.words("austen-emma.txt") if w.isalpha()]
n_stop = sum(1 for w in words if w in STOP)
print()
print(f"  Austen's Emma:  {len(words):,} word tokens, {n_stop:,} of them on the list")
print(f"                  that is {100 * n_stop / len(words):.0f}% of the text, gone in one pass.")


# ── the case against ─────────────────────────────────────────────────────
head("3. What else is on the list")

negations = ["not", "no", "nor", "against", "only", "very"]
for word in negations:
    print(f"    {word:9} on the list?  {word in STOP}")

print()
print("  Every one of those changes the meaning of a sentence, and 'not' is")
print("  the most important word English has for a negative judgement.")


head("4. The reversal")

for sentence in ["would not recommend",
                 "the battery is not very good",
                 "no problems at all"]:
    print(f"  before   {sentence}")
    print(f"  after    {strip_stop_words(sentence)}")
    print()

print("  A sentiment classifier reading the second line of each pair would call")
print("  all three of these positive. Nothing raised an error — the preprocessing")
print("  step simply destroyed the information the task depended on. This is the")
print("  same failure as VADER on 'nightmare' in lab 1.")


# ── the fix, when you must remove them ───────────────────────────────────
head("5. A curated list keeps the negations")

# The curated exception list: the negations and the contrast words, the ones
# whose removal flipped the sentences in section 4. Everything else on NLTK's
# list still goes. There is nothing official about this set — it is a judgement
# about which words carry the meaning of the task in front of us.
KEEP = {"not", "no", "nor", "never", "none", "against", "only", "very",
        "but", "however"}

for sentence in ["would not recommend", "the battery is not very good"]:
    print(f"  before        {sentence}")
    print(f"  NLTK list     {strip_stop_words(sentence)}")
    print(f"  curated       {strip_stop_words(sentence, keep=KEEP)}")
    print()

print("  The decision rule: remove them when you want the topic (search,")
print("  information retrieval — lab 4). Keep them when meaning depends on")
print("  them (sentiment, question answering). If you must remove them for a")
print("  meaning task, spare the negations first.")

print()
