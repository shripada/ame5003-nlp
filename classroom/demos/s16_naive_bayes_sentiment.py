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
Section 6 redoes the arithmetic in logs, which is how the classifier is
implemented in practice.

lessons/0016-naive-bayes-for-sentiment.html works the five-review corpus from
SLP3 Appendix B instead. Two corpora, one method; the arithmetic here is the
deck's.
"""

import math
from collections import Counter  # a dict that counts things: Counter["a", "a"] -> {"a": 2}


# ── the data ─────────────────────────────────────────────────────────────
#
# Each review is a plain string. We split it on spaces to get its words, so
# "great movie".split() gives ["great", "movie"].

# The training reviews we already know are positive.
POSITIVE = [
    "great movie",
    "great acting loved",
    "brilliant loved movie",
    "loved great film",
]

# The training reviews we already know are negative.
NEGATIVE = [
    "boring movie",
    "terrible acting hated",
    "boring hated film",
]

# The new review we want the model to classify. It is not in the training data.
REVIEW = "loved great movie"

# One name for each class, pointing at that class's reviews. Looping over
# CLASSES gives the labels "positive" and "negative" in that order.
CLASSES = {"positive": POSITIVE, "negative": NEGATIVE}


def head(title: str) -> None:
    """Print a section title, underlined with dashes."""
    print()                       # a blank line before each section
    print(title)
    print("-" * len(title))       # as many dashes as the title has characters


# ── training ─────────────────────────────────────────────────────────────
#
# Training a Naive Bayes classifier is counting, twice. Once over reviews,
# which gives the prior; once over words within a class, which gives the
# likelihoods.

# Step 1: the prior. P(class) = reviews in that class / all reviews.
total_reviews = len(POSITIVE) + len(NEGATIVE)     # 4 + 3 = 7

prior = {}                                        # label -> P(label)
for label, reviews in CLASSES.items():
    prior[label] = len(reviews) / total_reviews   # 4/7 and 3/7

# Step 2: count how often each word appears in each class.
# word_counts["positive"]["great"] will be the number of times "great"
# appears across all the positive reviews.
word_counts = {}
for label, reviews in CLASSES.items():
    counts = Counter()                  # starts empty; a missing word counts as 0
    for review in reviews:
        for word in review.split():     # "great movie" -> "great", "movie"
            counts[word] += 1           # one more sighting of this word
    word_counts[label] = counts

# The total number of words in each class (repeats included). This is the
# denominator of every likelihood for that class: 11 positive, 8 negative.
class_total = {}
for label, counts in word_counts.items():
    class_total[label] = sum(counts.values())

# The vocabulary is every distinct word seen in training, from either class.
# The vocabulary is pooled across both classes, and V is the same in both
# denominators even though the class totals differ. Add-one adds one count to
# every word the model could meet, not to every word this class happened to
# contain.
vocabulary_set = set()                  # a set keeps each word only once
for counts in word_counts.values():
    for word in counts:
        vocabulary_set.add(word)
VOCABULARY = sorted(vocabulary_set)     # sorted, so the table prints alphabetically
V = len(VOCABULARY)                     # the vocabulary size, 9


def likelihood(word: str, label: str) -> float:
    """P(word | class), add-one smoothed: (count + 1) / (class words + V)."""
    count = word_counts[label][word]    # 0 if the word never appeared in this class
    return (count + 1) / (class_total[label] + V)


def raw_likelihood(word: str, label: str) -> float:
    """P(word | class) with no smoothing — a word's plain share of its class."""
    count = word_counts[label][word]
    return count / class_total[label]   # is exactly 0 for an unseen word


head("1. The training corpus, and the prior")

# Print each class's reviews, one per line.
for label, reviews in CLASSES.items():
    print(f"    {label.upper():8} ({len(reviews)} reviews)")   # :8 pads to 8 characters
    for review in reviews:
        print(f"        {review}")

# :.2f prints a number with two decimal places.
print(f"""
    P(positive) = {len(POSITIVE)}/{total_reviews} = {prior['positive']:.2f}
    P(negative) = {len(NEGATIVE)}/{total_reviews} = {prior['negative']:.2f}

  That is the whole of the first training step: count the reviews. The corpus
  leans positive, {len(POSITIVE)} against {len(NEGATIVE)}, so the model tilts that way before it has read
  a single word — and that tilt is carried into every prediction it makes.

  A review is a bag of words from here on. Which words appear, not the order
  they appear in.""")


head("2. Count every word, once per class")

# A table: one row per vocabulary word, one column per class.
# :>12 right-aligns a value in a column 12 characters wide.
print(f"    {'word':12} {'in POSITIVE':>12} {'in NEGATIVE':>12}")
for word in VOCABULARY:
    positive_count = word_counts["positive"][word]
    negative_count = word_counts["negative"][word]
    print(f"    {word:12} {positive_count:>12} {negative_count:>12}")
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

# Show each class's denominator: its word total plus the vocabulary size.
for label in CLASSES:
    denominator = class_total[label] + V        # 11 + 9 = 20, and 8 + 9 = 17
    print(f"    {label:8} denominator = {class_total[label]} + {V} "
          f"= {denominator}")

# For each word of the new review, show the smoothed likelihood in each class,
# with the working written out: (count+1)/denominator = answer.
print(f"\n    {'word':12} {'P(w | pos)':>18} {'P(w | neg)':>18}")
for word in REVIEW.split():
    cells = []                                  # one piece of text per class
    for label in CLASSES:
        count = word_counts[label][word]
        denominator = class_total[label] + V
        answer = likelihood(word, label)
        cells.append(f"({count}+1)/{denominator} = {answer:.3f}")
    positive_cell = cells[0]                    # CLASSES lists positive first
    negative_cell = cells[1]
    print(f"    {word:12} {positive_cell:>18} {negative_cell:>18}")

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
    """Prior x the likelihood of each known word. The Bayes numerator.

    smoothed=True uses the add-one likelihoods; smoothed=False uses the raw
    ones, which section 5 needs to show what goes wrong without the +1.
    """
    product = prior[label]                      # start from P(class)
    for word in review.split():
        if word not in VOCABULARY:
            continue                            # an unknown word is skipped
        if smoothed:
            product = product * likelihood(word, label)
        else:
            product = product * raw_likelihood(word, label)
    return product


def classify(review: str, smoothed: bool = True) -> dict[str, float]:
    """Posterior per class: each score divided by the evidence, their sum."""
    # The score of the review under each class.
    scores = {}
    for label in CLASSES:
        scores[label] = score(review, label, smoothed)

    # The evidence P(review) is the sum of those scores.
    evidence = sum(scores.values())

    # If every score is zero there is nothing to divide by, so there is no
    # answer. nan ("not a number") says exactly that.
    if evidence == 0:
        return {label: float("nan") for label in scores}

    # Divide each score by the evidence, so the results add up to 1.
    posterior = {}
    for label, value in scores.items():
        posterior[label] = value / evidence
    return posterior


head("4. Score a review the model has never seen")

words = REVIEW.split()
print(f"    review: {REVIEW!r}   ->   words: {', '.join(words)}\n")

# Write out the multiplication for each class, e.g.
#   0.57 x 0.200 x 0.200 x 0.150 = 0.00343
for label in CLASSES:
    factors = []                                # the likelihoods as text
    for word in words:
        factors.append(f"{likelihood(word, label):.3f}")
    working = " x ".join(factors)               # "0.200 x 0.200 x 0.150"
    print(f"    if {label.upper():8}  {prior[label]:.2f} x {working} "
          f"= {score(REVIEW, label):.5f}")

positive_score = score(REVIEW, "positive")
negative_score = score(REVIEW, "negative")
evidence = positive_score + negative_score      # the Bayes denominator
posterior = classify(REVIEW)                    # the two scores, divided by it

print(f"\n    evidence = {positive_score:.5f} + "
      f"{negative_score:.5f} = {evidence:.5f}\n")
for label in CLASSES:
    # :.0% prints a fraction as a whole-number percentage: 0.953 -> 95%
    print(f"    P({label} | review) = {score(REVIEW, label):.5f} / "
          f"{evidence:.5f} = {posterior[label]:.0%}")

# The verdict is the class with the larger posterior.
if posterior["positive"] >= posterior["negative"]:
    verdict = "positive"
else:
    verdict = "negative"

print(f"""
    verdict: {verdict.upper()}, with {posterior[verdict]:.0%} confidence.

  Every Bayes term appeared exactly once: the prior, a likelihood per word, an
  evidence to divide by, a posterior to compare. The two scores are tiny and
  neither is a probability on its own; the division is what turns them into
  one.""")


head("5. The same review, without the +1")

# The raw likelihoods: a word's count divided by its class total, no +1.
print(f"    {'word':12} {'raw P(w | pos)':>16} {'raw P(w | neg)':>16}")
for word in REVIEW.split():
    raw_positive = raw_likelihood(word, "positive")
    raw_negative = raw_likelihood(word, "negative")
    print(f"    {word:12} {raw_positive:>16.3f} {raw_negative:>16.3f}")

# Score the review again, this time without smoothing.
unsmoothed = {}
for label in CLASSES:
    unsmoothed[label] = score(REVIEW, label, smoothed=False)

print(f"""
    score if positive = {unsmoothed['positive']:.5f}
    score if negative = {unsmoothed['negative']:.5f}

  'loved' never appears in a negative review, so its raw share is {word_counts['negative']['loved']}/{class_total['negative']} = 0, and
  one zero in the product zeroes the whole class. The negative class has been
  eliminated by a single word, and the other two words never got a vote —
  including 'movie', which the negative reviews really do contain.""")

# A review with words from both sides: each side has a word the other never saw.
MIXED = "boring hated great"

raw = {}                                        # unsmoothed scores
for label in CLASSES:
    raw[label] = score(MIXED, label, smoothed=False)
smooth = classify(MIXED)                        # smoothed posteriors

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


head("6. Working in logs")

# Multiplying many numbers below 1 gives a smaller number each time. A
# computer float cannot go below about 1e-308; anything smaller is rounded
# to exactly 0.0, which is called underflow. Logarithms avoid it because
#
#     log(a x b x c) = log a + log b + log c
#
# so instead of multiplying tiny probabilities we add their logs. A log of a
# number below 1 is negative, and adding negatives just gives a larger
# negative number, which a float holds with no trouble.


def log_score(review: str, label: str) -> float:
    """The score as a sum of logs: log P(class) + log P(word | class) + ...

    This is the log of score(review, label), computed without ever forming
    the tiny product itself.
    """
    total = math.log(prior[label])              # start from log P(class)
    for word in review.split():
        if word not in VOCABULARY:
            continue                            # unknown words are skipped, as before
        total = total + math.log(likelihood(word, label))
    return total


# 6a. The same sum as section 4, written out in logs.
print(f"    review: {REVIEW!r}\n")
for label in CLASSES:
    terms = [f"{math.log(prior[label]):.3f}"]   # the log prior comes first
    for word in REVIEW.split():
        terms.append(f"{math.log(likelihood(word, label)):.3f}")
    working = " + ".join(terms)                 # e.g. "-0.560 + -1.609 + ..."
    total = log_score(REVIEW, label)
    # math.exp undoes math.log, so exp(log score) must give back the score.
    print(f"    if {label.upper():8}  {working} = {total:.3f}"
          f"   ->  exp = {math.exp(total):.5f}")

print("""
  Each log is a modest negative number, and the sum is too. Raising e to the
  sum gives back exactly the scores of section 4, so nothing has been lost:
  the log score is the same score, written on a different scale. Because log
  only ever grows as its input grows, the class with the larger score also has
  the larger log score, and the verdict is unchanged.""")

# 6b. Longer and longer reviews: the product underflows, the log sum does not.
# We repeat the review to make it longer: [a, b] * 2 = [a, b, a, b].
print(f"\n    {'words':>6} {'product pos':>12} {'product neg':>12}"
      f" {'log pos':>10} {'log neg':>10}   verdict from logs")
for repeats in [1, 10, 100, 300]:
    long_words = REVIEW.split() * repeats
    long_review = " ".join(long_words)          # back into one string

    product_pos = score(long_review, "positive")
    product_neg = score(long_review, "negative")
    log_pos = log_score(long_review, "positive")
    log_neg = log_score(long_review, "negative")

    # Compare the log scores: the larger (less negative) one wins.
    if log_pos >= log_neg:
        log_verdict = "positive"
    else:
        log_verdict = "negative"

    # :.3g prints a number in its shortest readable form, e.g. 1.2e-300
    print(f"    {len(long_words):>6} {product_pos:>12.3g} {product_neg:>12.3g}"
          f" {log_pos:>10.1f} {log_neg:>10.1f}   {log_verdict}")

print("""
  At 900 words both products have underflowed to exactly 0, and comparing 0
  with 0 tells us nothing. The log scores are ordinary negative numbers
  and still say which class is ahead.""")

# 6c. Getting the percentage back without ever leaving the log scale.
# The posterior is score / (sum of scores). We cannot compute the scores
# themselves (they underflow), but we can shift every log score by the same
# amount first: subtracting the largest log score divides every score by the
# same number, which cancels in the division. The winner then has log score
# 0, i.e. score 1, and nothing is small enough to underflow any more. This
# trick is known as log-sum-exp.
long_review = " ".join(REVIEW.split() * 300)
logs = {}
for label in CLASSES:
    logs[label] = log_score(long_review, label)

largest = max(logs.values())                    # the winner's log score
shifted = {}
for label, value in logs.items():
    shifted[label] = math.exp(value - largest)  # 1 for the winner, smaller for the rest

total = sum(shifted.values())
print(f"\n    the 900-word review, normalised from its log scores:")
for label in CLASSES:
    print(f"    P({label} | review) = {shifted[label] / total:.0%}")

print("""
  The product method could give no answer at all for this review. Working in
  logs gives a verdict and a confidence, and for a review this long the
  confidence is close to certain because the same evidence has been counted
  three hundred times. This is how Naive Bayes is implemented in practice:
  train with counts, predict with sums of logs.""")

head("7. What it cannot do")

# Four test reviews, each chosen to expose a weakness.
TEST_REVIEWS = ["great movie", "not great movie", "great terrible movie",
                "superb cinematography"]

for review in TEST_REVIEWS:
    # Sort the words into those the model knows and those it will drop.
    known = []
    dropped = []
    for word in review.split():
        if word in VOCABULARY:
            known.append(word)
        else:
            dropped.append(word)

    result = classify(review)                   # the posterior for each class
    # The label with the larger posterior wins, as in section 4.
    if result["positive"] >= result["negative"]:
        label = "positive"
    else:
        label = "negative"

    kept_text = ", ".join(known) or "nothing"   # "nothing" if no word was known
    note = ""
    if dropped:
        note = f"   (dropped: {', '.join(dropped)})"
    print(f"    {review:24} -> {label:8} {result[label]:>5.0%}"
          f"   kept: {kept_text}{note}")

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
