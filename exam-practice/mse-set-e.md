# AME 5003 — Principles of NLP
## Mid-semester examination — Sample paper, Set E

**Duration: 1 hour 50 minutes.  Maximum marks: 50.**

Answer **all** questions.

---

## Question 1 — Text processing  (10 marks)

**(a)  Consider the following text:**  *(5 marks)*

```
Workshop 8 (Room 204) begins at 11:45 (sharp).
```

  1. Apply a regular expression that captures the hours and the minutes of `11:45` as two
     groups.  *(2 marks)*
  2. The pattern `\(.*\)` is used to extract `(Room 204)`. State what it matches on this
     sentence, explain why, and correct the pattern.  *(3 marks)*

**(b)  Normalisation and stemming**  *(5 marks)*

A search index lower-cases every token and applies the Porter stemmer, which maps both
*organ* and *organism* to `organ`.

  1. Explain how this affects the results of the query *organ*, and say whether a lemmatizer
     would behave differently.  *(3 marks)*
  2. Of the tokens `RAM`, `ram` and `R.A.M.`, say which should be treated as one term, and
     explain whether lower-casing achieves that.  *(2 marks)*

---

## Question 2 — Linguistic processing and retrieval  (10 marks)

**(a)  Part-of-speech tagging and named entities**  *(5 marks)*

```
Kavya Menon sent a letter to South Africa on Wednesday .
```

  1. Tag every token with the BIO scheme, using the types PERSON, GPE and DATE.
     *(3 marks)*
  2. Give the POS tag of *address* in `Address the audience` and in `Check the address`, and
     explain from these two sentences why a tagger cannot assign tags by looking words up.
     *(2 marks)*

**(b)  The inverted index and TF-IDF**  *(5 marks)*

```
D1  solar power grid     D3  solar grid battery
D2  power battery        D4  power grid battery
```

  1. Write the postings lists for `power` and `grid`, and answer `power AND grid` by merging
     them.  *(2 marks)*
  2. In a collection of N = 1,000,000 documents, `battery` has df = 10,000 and occurs 100
     times in a document. Compute its weight with tf-idf = (1 + log₁₀ tf) × log₁₀(N/df), and
     state its weight if it occurred in every document.  *(3 marks)*

---

## Question 3 — N-gram probabilities  (10 marks)

A bigram model is trained on this corpus:

```
<s> maya reads books </s>
<s> maya reads news </s>
<s> omar reads books </s>
<s> omar writes news </s>
```

**(a)  Estimating a sentence probability**  *(5 marks)*

  1. Write P(omar reads news) as a product of bigram probabilities, including both boundary
     symbols, and evaluate it.  *(3 marks)*
  2. Estimate P(news | `<s>` omar reads) from the corpus, using the full history. Compare it
     with the bigram estimate P(news | reads), and explain what this shows about the
     assumption the bigram model makes.  *(2 marks)*

**(b)  The zero problem and add-one smoothing**  *(5 marks)*

  1. Compute P(maya writes books). State the value and name the problem it shows.
     *(2 marks)*
  2. Recompute P(writes | maya) with add-one smoothing, stating V. Then compute
     P(reads | maya) before and after smoothing, and explain what the change shows about
     add-one smoothing.  *(3 marks)*

---

## Question 4 — Interpolation and perplexity  (10 marks)

**(a)  Interpolation and backoff**  *(5 marks)*

For the word *news*, a model has these estimates:

```
P(news | maya writes) = 0     P(news | writes) = 0.6     P(news) = 0.2
```

  1. Compute the interpolated probability with λ₁ = 0.6, λ₂ = 0.3, λ₃ = 0.1.
     *(2 marks)*
  2. Give the estimate backoff would use here and compare it with your answer to 1. Explain
     why the λ values should not be chosen on the training corpus.  *(3 marks)*

**(b)  Perplexity**  *(5 marks)*

A bigram model assigns the test sentence `<s> maya writes news </s>` the probability 1/256.

  1. State N for this sentence and compute the perplexity.  *(3 marks)*
  2. A model that treats all 7 tokens of the vocabulary as equally likely is scored on the same
     sentence. Compute its perplexity and explain what the comparison tells us.
     *(2 marks)*

---

## Question 5 — Tokenisation  (10 marks)

**(a)  Rule-based word tokenisation**  *(5 marks)*

Consider the following sentence:

```
Ms. Roy isn't waiting.
```

  1. Apply tokenisation by splitting only at spaces, and write the resulting sequence of
     tokens.  *(2 marks)*
  2. Write the tokens a Penn Treebank-style rule-based tokeniser should produce. Explain why it
     keeps the full stop in `Ms.` but separates the final full stop, and why it splits `isn't`.
     *(3 marks)*

**(b)  Byte-Pair Encoding**  *(5 marks)*

The number is the word's frequency; `_` marks the end of a word. Break ties by taking the
left-hand pair.

```
5   r u n _
4   f u n _
3   r u n s _
```

  1. Perform **two** BPE merges, stating the pair and its count for each. Then apply the merges
     in order to segment the unseen word `f u n s _`.  *(3 marks)*
  2. Explain why the resulting BPE vocabulary can represent *funs* without `<UNK>`, and state
     one effect of increasing the number of merges.  *(2 marks)*

---

*End of question paper.*
