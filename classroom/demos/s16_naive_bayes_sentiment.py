#!/usr/bin/env python3
"""Session 16 — a sentiment analyser trained by hand, counts to verdict.

    uv run demos/s16_naive_bayes_sentiment.py

The corpus is the seven movie reviews from the PowerPoint deck
slides/0016-naive-bayes-for-sentiment.pptx, so every number printed here is
one the class has just seen on a slide: the 4/7 prior, the 20 and 17
denominators, the two scores 0.00343 and 0.00017, and the 95% verdict. They
are recomputed from the reviews rather than quoted, which is the point of
running it — change a review at the top and watch the verdict move.

Section 5 is the one to linger on: the same review is scored again with the
smoothing removed, and a single unseen word vetoes an entire class.

lessons/0016-naive-bayes-for-sentiment.html works the five-review corpus from
SLP3 Appendix B instead. Two corpora, one method; the arithmetic here is the
deck's.
"""

import collections
import math

POSITIVE = [
    "great movie",
    "great acting loved",
    "brilliant loved movie",
    "loved great film",
]
NEGATIVE = [
    "boring movie",
    "terrible acting hated",
    "boring hated film",
]

REVIEW = "loved great movie"

CLASSES = {"positive": POSITIVE, "negative": NEGATIVE}


def head(title: str) -> None:
    print(f"\n{title}\n{'-' * len(title)}")


# ── training ─────────────────────────────────────────────────────────────
#
# Training a Naive Bayes classifier is counting, twice. Once over reviews,
# which gives the prior; once over words within a class, which gives the
# likelihoods. Nothing is fitted, nothing is iterated.

n_reviews = sum(len(reviews) for reviews in CLASSES.values())
prior = {label: len(reviews) / n_reviews for label, reviews in CLASSES.items()}

word_counts = {
    label: collections.Counter(word for review in reviews for word in review.split())
    for label, reviews in CLASSES.items()
}
class_total = {label: sum(counts.values()) for label, counts in word_counts.items()}

# The vocabulary is pooled across both classes, and V is the same in both
# denominators even though the class totals differ. Add-one adds one count to
# every word the model could meet, not to every word this class happened to
# contain.
VOCABULARY = sorted(set().union(*word_counts.values()))
V = len(VOCABULARY)


def likelihood(word: str, label: str) -> float:
    """P(word | class), add-one smoothed."""
    return (word_counts[label][word] + 1) / (class_total[label] + V)


def raw_likelihood(word: str, label: str) -> float:
    """P(word | class) with no smoothing — a word's plain share of its class."""
    return word_counts[label][word] / class_total[label]


head("1. The training corpus, and the prior")

for label, reviews in CLASSES.items():
    print(f"    {label.upper():8} ({len(reviews)} reviews)")
    for review in reviews:
        print(f"        {review}")

print(f"""
    P(positive) = {len(POSITIVE)}/{n_reviews} = {prior['positive']:.2f}
    P(negative) = {len(NEGATIVE)}/{n_reviews} = {prior['negative']:.2f}

  That is the whole of the first training step: count the reviews. The corpus
  leans positive, {len(POSITIVE)} against {len(NEGATIVE)}, so the model tilts that way before it has read
  a single word — and that tilt is carried into every prediction it makes.

  A review is a bag of words from here on. Which words appear, not the order
  they appear in.""")


head("2. Count every word, once per class")

print(f"    {'word':12} {'in POSITIVE':>12} {'in NEGATIVE':>12}")
for word in VOCABULARY:
    print(f"    {word:12} {word_counts['positive'][word]:>12} "
          f"{word_counts['negative'][word]:>12}")
print(f"    {'TOTAL words':12} {class_total['positive']:>12} "
      f"{class_total['negative']:>12}")

print(f"""
  {V} distinct words, and the two class totals differ ({class_total['positive']} against {class_total['negative']}) because the
  positive reviews are both more numerous and slightly longer. Nothing forces
  them to match.

  Notice the zeros. 'great' never appears in a negative review, 'boring' never
  in a positive one. Those zeros are the reason step 3 is not simply
  division.""")


head("3. Counts into likelihoods, with the +1")

print("    P(word | class) = (count + 1) / (class words + V)\n")
for label in CLASSES:
    print(f"    {label:8} denominator = {class_total[label]} + {V} "
          f"= {class_total[label] + V}")

print(f"\n    {'word':12} {'P(w | pos)':>18} {'P(w | neg)':>18}")
for word in REVIEW.split():
    cells = []
    for label in CLASSES:
        count, denominator = word_counts[label][word], class_total[label] + V
        cells.append(f"({count}+1)/{denominator} = {likelihood(word, label):.3f}")
    print(f"    {word:12} {cells[0]:>18} {cells[1]:>18}")

print("""
  The prior and one likelihood per word per class: that is the entire trained
  model. There is nothing else stored, and training will not be mentioned
  again.""")


# ── predicting ───────────────────────────────────────────────────────────
#
# Prediction is the Bayes numerator, once per class, then a normalisation.
# The vocabulary check is the standard treatment of an unknown word: it is
# dropped, because it multiplies every class by the same factor and so cannot
# change which class wins.

def score(review: str, label: str, smoothed: bool = True) -> float:
    """Prior x the likelihood of each known word. The Bayes numerator."""
    estimate = likelihood if smoothed else raw_likelihood
    product = prior[label]
    for word in review.split():
        if word in VOCABULARY:
            product *= estimate(word, label)
    return product


def classify(review: str, smoothed: bool = True) -> dict[str, float]:
    """Posterior per class: each score divided by the evidence, their sum."""
    scores = {label: score(review, label, smoothed) for label in CLASSES}
    evidence = sum(scores.values())
    if evidence == 0:
        return {label: float("nan") for label in scores}
    return {label: value / evidence for label, value in scores.items()}


head("4. Score a review the model has never seen")

print(f"    review: {REVIEW!r}   ->   words: {', '.join(REVIEW.split())}\n")
for label in CLASSES:
    factors = " x ".join(f"{likelihood(word, label):.3f}"
                         for word in REVIEW.split())
    print(f"    if {label.upper():8}  {prior[label]:.2f} x {factors} "
          f"= {score(REVIEW, label):.5f}")

evidence = sum(score(REVIEW, label) for label in CLASSES)
posterior = classify(REVIEW)
print(f"\n    evidence = {score(REVIEW, 'positive'):.5f} + "
      f"{score(REVIEW, 'negative'):.5f} = {evidence:.5f}\n")
for label in CLASSES:
    print(f"    P({label} | review) = {score(REVIEW, label):.5f} / "
          f"{evidence:.5f} = {posterior[label]:.0%}")

verdict = max(posterior, key=posterior.get)
print(f"""
    verdict: {verdict.upper()}, with {posterior[verdict]:.0%} confidence.

  Every Bayes term appeared exactly once: the prior, a likelihood per word, an
  evidence to divide by, a posterior to compare. The two scores are tiny and
  neither is a probability on its own; the division is what turns them into
  one.""")


head("5. The same review, without the +1")

print(f"    {'word':12} {'raw P(w | pos)':>16} {'raw P(w | neg)':>16}")
for word in REVIEW.split():
    print(f"    {word:12} {raw_likelihood(word, 'positive'):>16.3f} "
          f"{raw_likelihood(word, 'negative'):>16.3f}")

unsmoothed = {label: score(REVIEW, label, smoothed=False) for label in CLASSES}
print(f"""
    score if positive = {unsmoothed['positive']:.5f}
    score if negative = {unsmoothed['negative']:.5f}

  'loved' never appears in a negative review, so its raw share is {word_counts['negative']['loved']}/{class_total['negative']} = 0, and
  one zero in the product zeroes the whole class. The negative class has been
  eliminated by a single word, and the other two words never got a vote —
  including 'movie', which the negative reviews really do contain.""")

MIXED = "boring hated great"
raw = {label: score(MIXED, label, smoothed=False) for label in CLASSES}
smooth = classify(MIXED)
print(f"""
  Here the surviving class is the right one and the verdict happens to hold.
  That is luck. Try {MIXED!r}:

    without the +1   score if positive = {raw['positive']:.5f}   score if negative = {raw['negative']:.5f}
    with the +1      P(positive) = {smooth['positive']:.0%}   P(negative) = {smooth['negative']:.0%}

  'boring' vetoes the positive class and 'great' vetoes the negative one, so
  both scores are zero and there is nothing to divide by: no verdict at all.
  With the +1 the two clear negative words outvote the one positive word.
  Unseen words are the normal case in real text, so this is not an edge case
  to be handled later; smoothing is what makes the method usable at all.""")


head("6. Underflow, and what the log fixes")

for repeats in [1, 100, 300]:
    long_review = " ".join(REVIEW.split() * repeats)
    words = len(long_review.split())
    print(f"    a {words:>4}-word review  ->  score "
          f"{score(long_review, verdict):.3g}")

log_scores = {
    label: math.log(prior[label]) + sum(math.log(likelihood(word, label))
                                        for word in REVIEW.split())
    for label in CLASSES
}
print(f"""
  Multiplying a few hundred numbers below 1 drives the score towards the
  smallest float there is, and a review of any real length runs past it: at
  900 words the score is exactly 0.0, and so is every class's, which makes the
  comparison meaningless. Add the logarithms instead:

    log score if positive = {log_scores['positive']:.3f}
    log score if negative = {log_scores['negative']:.3f}

  The numbers are comfortable, the larger is still the larger, and the
  ordering is what the decision needs. Nothing else about the method changes.""")


head("7. What it cannot do")

for review in ["great movie", "not great movie", "great terrible movie",
               "superb cinematography"]:
    known = [w for w in review.split() if w in VOCABULARY]
    dropped = [w for w in review.split() if w not in VOCABULARY]
    result = classify(review)
    label = max(result, key=result.get)
    note = f"   (dropped: {', '.join(dropped)})" if dropped else ""
    print(f"    {review:24} -> {label:8} {result[label]:>5.0%}"
          f"   kept: {', '.join(known) or 'nothing'}{note}")

print("""
  Three failures, in order. 'not' is not in the vocabulary and is dropped, so
  a negation reads exactly like the praise it reverses. 'great terrible movie'
  has one word pulling each way and the model multiplies both in, having no
  notion that the pair means something the two words separately do not. And a review of words it has never
  met keeps the prior alone, which is a guess dressed as a decision.

  All three come from the same two assumptions: bag of words throws away
  order, and independence throws away interaction. Both are false and the
  classifier works anyway, which is worth sitting with. The models later in
  the course fix these by reading a word in its context — a more elaborate
  version of the job just done here by counting.""")
