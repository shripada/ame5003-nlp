# AME 5003 — Principles of NLP
## Mid-semester examination — Sample paper, Set A

**Duration: 1 hour 50 minutes.  Maximum marks: 50.**

Answer **all** questions.

---

## Question 1 — Text processing  (10 marks)

**(a)  Consider the following text:**  *(5 marks)*

```
Pay Rs 450 (cash) or Rs 430 (UPI) by 17:30.
```
  
  1. Apply a regular expression that captures the hours and the minutes of `17:30` as two
     groups.  *(2 marks)*
  2. The pattern `\(.*\)` is used to extract `(cash)`. State what it matches on this sentence,
     explain why, and correct the pattern.  *(3 marks)*

**(b)  Normalisation and stemming**  *(5 marks)*

A search index lower-cases every token and applies the Porter stemmer, which maps both
*university* and *universe* to `univers`.

  1. Explain how this affects the results of the query *university*, and say whether a
     lemmatizer would behave differently.  *(3 marks)*
  2. Of the tokens `WHO`, `who` and `W.H.O.`, say which should be treated as one term, and
     explain whether lower-casing achieves that.  *(2 marks)*

---

## Question 2 — Linguistic processing and retrieval  (10 marks)

**(a)  Part-of-speech tagging and named entities**  *(5 marks)*

```
Priya Nair booked a flight to New Delhi on Monday .
```

  1. Tag every token with the BIO scheme, using the types PERSON, GPE and DATE.
     *(3 marks)*
  2. Give the POS tag of *book* in `Book a flight` and in `Read the book`, and explain from
     these two sentences why a tagger cannot assign tags by looking words up.
     *(2 marks)*

**(b)  The inverted index and TF-IDF**  *(5 marks)*

```
D1  rain flood road      D3  rain road
D2  flood relief         D4  flood road relief
```

  1. Write the postings lists for `flood` and `road`, and answer `flood AND road` by merging
     them.  *(2 marks)*
  2. In a collection of N = 10,000 documents, `relief` has df = 100 and occurs 10 times in a
     document. Compute its weight with tf-idf = (1 + log₁₀ tf) × log₁₀(N/df), and state its
     weight if it occurred in every document.  *(3 marks)*

---

## Question 3 — N-gram probabilities  (10 marks)

A bigram model is trained on this corpus:

```
<s> we like tea </s>
<s> they like coffee </s>
<s> we drink coffee </s>
```

**(a)  Estimating a sentence probability**  *(5 marks)*

  1. Write P(we like coffee) as a product of bigram probabilities, including both boundary
     symbols, and evaluate it.  *(3 marks)*
  2. Estimate P(coffee | `<s>` we like) from the corpus, using the full history. Compare it
     with the bigram estimate P(coffee | like), and explain what this shows about the
     assumption the bigram model makes.  *(2 marks)*

**(b)  The zero problem and add-one smoothing**  *(5 marks)*

  1. Compute P(they drink tea). State the value and name the problem it shows.
     *(2 marks)*
  2. Recompute P(drink | they) with add-one smoothing, stating V. Then compute P(like | they)
     before and after smoothing, and explain what the change shows about add-one smoothing.
     *(3 marks)*

---

## Question 4 — Interpolation and perplexity  (10 marks)

**(a)  Interpolation and backoff**  *(5 marks)*

For the word *tea*, a model has these estimates:

```
P(tea | they like) = 0     P(tea | like) = 0.5     P(tea) = 0.2
```

  1. Compute the interpolated probability with λ₁ = 0.6, λ₂ = 0.3, λ₃ = 0.1.
     *(2 marks)*
  2. Give the estimate backoff would use here and compare it with your answer to 1. Explain
     why the λ values should not be chosen on the training corpus.  *(3 marks)*

**(b)  Perplexity**  *(5 marks)*

A bigram model assigns the test sentence `<s> they like tea </s>` the probability 1/81.

  1. State N for this sentence and compute the perplexity.  *(3 marks)*
  2. A model that treats all 7 tokens of the vocabulary as equally likely is scored on the same
     sentence. Compute its perplexity and explain what the comparison tells us.
     *(2 marks)*

---

## Question 5 — Tokenisation  (10 marks)

**(a)  Rule-based word tokenisation**  *(5 marks)*

Consider the following sentence:

```
Dr. Rao isn't ready.
```

  1. Apply tokenisation by splitting only at spaces, and write the resulting sequence of
     tokens.  *(2 marks)*
  2. Write the tokens a Penn Treebank-style rule-based tokeniser should produce. Explain why it
     keeps the full stop in `Dr.` but separates the final full stop, and why it splits `isn't`.
     *(3 marks)*

**(b)  Byte-Pair Encoding**  *(5 marks)*

The number is the word's frequency; `_` marks the end of a word.

```
5   h u g _
2   p u g _
3   h u g s _
```

  1. Perform **two** BPE merges, stating the pair and its count for each. Then apply the merges
     in order to segment the unseen word `p u g s _`.  *(3 marks)*
  2. Explain why the resulting BPE vocabulary can represent *pugs* without `<UNK>`, and state
     one effect of increasing the number of merges.  *(2 marks)*

---

*End of question paper.*
