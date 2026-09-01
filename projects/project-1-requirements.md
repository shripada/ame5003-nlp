# Project 1 — a mini search engine: requirements

This is the project written out as requirements, in plain English, for building
from an empty directory. It says what your system has to do and how it is
marked. It contains no code.

There is also a starter repository, with the corpus loading, a command line and
a test suite already written, and stubs for the parts you fill in. **The two
routes are marked identically.** Take the starter if you would rather spend your
time on the measuring and the writing, which is where most of the marks are.
Take this document if you would rather build the whole thing yourself. Neither
is the harder option on paper; building from scratch simply means you also write
the plumbing and the tests.

Read [the project brief](project-1-mini-search-engine.html) as well. It explains
*why* each stage is there and which session it comes from. This document is the
checklist; the brief is the reasoning.

---

## What you are building

A search engine over a collection of 10,788 news articles. Someone types a query
into a box in a browser. Your system returns the documents that best match, in
order, with the matching words marked, in well under a second.

Behind that box sits everything Unit I taught: a regular expression that pulls
structure out of raw text, a preprocessing pipeline, an inverted index, and
TF-IDF ranking. You are not asked to invent anything. Every component was
covered in a session and most were built in a lab, on six documents. The project
is what happens when you assemble them, point them at a real collection, and
then answer the question a working system always raises: which of my choices
actually helped?

Work through the stages in order. Each one produces something you can run and
check before you start the next. **Stage 6 is the interface. Do not start
there.** A search box over an index that returns the wrong documents is worth
very little, and the interface carries 10 marks of 100.

---

## The collection

You will index the Reuters-21578 corpus, a standard collection of Reuters
newswire articles from 1987. It comes with NLTK, so there is nothing to download
by hand — ask NLTK for the `reuters` corpus and it fetches it once.

Three things about it matter.

**Each document is a headline in capitals on its own first line, then a body.**
The body is hard-wrapped mid-sentence, the way a 1987 wire service wrapped it.
That headline is a piece of structure sitting inside unstructured text, which is
what stage 1 is about.

**Every document carries one or more topic labels**, assigned by Reuters — from
a set of 90 including `crude`, `grain`, `earn` and `acq`. Your search engine
must never read them. They exist in this project for one purpose: in stage 5
they let you check whether a change improved your results or made them worse,
instead of asserting that it helped.

**Everyone indexes the same documents in the same order.** Sort the corpus file
identifiers alphabetically and number them from zero, so document 0 is the first
of that sorted list. Do this and a document number means the same thing in your
submission as in everyone else's, and a claim about document 4174 can be
checked. Get it wrong and none of the numbers below will match yours.

---

## Ground rules

**Libraries you may use.** NLTK, spaCy, the `regex` module, NumPy, Hugging
Face's `tokenizers`, Streamlit, and anything in the Python standard library.

**What you must write yourself.** The inverted index, the postings list merges,
the TF-IDF weighting and the ranking. Not `TfidfVectorizer`, not Whoosh, not
Elasticsearch, not a vector database.

Scikit-learn is allowed for *checking* your work — comparing your weights
against `TfidfVectorizer` is a reasonable thing to do, and lab 7 showed the two
disagree — but the engine you submit must be yours.

**Python 3.10 or later.**

**Pin your dependency versions.** A report whose numbers cannot be reproduced
loses marks, and an unpinned NLTK is one of the ways that happens.

---

## Stage 1 — Pulling structure out of raw text

*Sessions 1 and 2. Lab 2 did this on bank SMS.*

Write something that takes one raw document and returns its parts, using regular
expressions:

1. **The headline** — the first line, with surrounding spaces removed.
2. **The body** — everything after the first line, with the hard wrapping
   collapsed so that it is one continuous line. A search result has to show a
   readable snippet, and you cannot take a readable snippet out of text that
   still has newlines in the middle of sentences.
3. **Money and quantity amounts.** The corpus is full of them and they come in
   several shapes: `3.5 mln dlrs`, `1,250,000`, `2.1 pct`, `500 tonnes`. Catch
   the common ones. Report how many of the first 500 documents contain at least
   one — a reasonable expression finds them in more than 300.
4. **Dates.** The corpus writes them as `March 5`, `MARCH 1987`, `5 MARCH 1987`
   and other variants. Catch what you reasonably can.

You will not catch everything and you are not expected to. What you are expected
to do is **look at what you missed**. Pick twenty documents at random, read your
own output against the text, and record in your report one thing your expression
gets wrong and why. Session 2's greedy trap is a good place to start looking.

Regular expressions are for this stage only. Do not use one as your tokenizer —
session 3 explained at length why that fails, and stage 2 is where you use a real
one.

---

## Stage 2 — The preprocessing pipeline

*Sessions 3, 4 and 5. Lab 3 built these steps one at a time.*

Write **one** thing — a function with keyword arguments, or a small class, your
choice — that turns a piece of text into a list of index terms, with each step
switchable on and off. The steps, in this order:

1. **Normalise the Unicode**, NFKC, on the whole string, always. Do this
   *before* tokenizing: a compatibility ligature has to become ordinary letters
   before anything can split the text on word boundaries. This step is not
   optional and is not tied to any switch.
2. **Tokenize** with NLTK's word tokenizer or spaCy's. Not by splitting on
   spaces, and not with a regular expression.
3. **Keep only the tokens that are entirely letters.** That discards punctuation
   and bare numerals. The numbers you pulled out in stage 1 are metadata, not
   index terms.
4. **Case fold**, if that switch is on.
5. **Remove stop words**, if that switch is on, using NLTK's English list.
   Compare the lowercased token against the list whether or not case folding is
   on — the list is lowercase, so testing an unfolded token would silently keep
   every capitalised `The`.
6. **Stem** with Porter, **or** lemmatize with WordNet or spaCy, or neither.
   Asking for both at once is a mistake and your code should refuse it.

Return the terms in the order they appeared, with repeats kept in. Stage 4 counts
them.

### The rule this project is built around

**The query must go through exactly the same pipeline as the documents, with the
same switches set.**

If your index holds `opec` because the pipeline folded case, and someone's
`OPEC` reaches the index unfolded, the term is simply absent and the query
silently returns nothing at all. Session 9 said this. It is the single most
common way this project goes wrong, and it fails quietly — you get an empty
result page and no error.

The way to make this impossible rather than merely unlikely is to have the index
hold on to the pipeline it was built with, and use that one for every query.
Then there is nothing to remember at the call site, because there is no second
place where the settings could be chosen.

There is a second reason this has to be switchable: stage 5 runs the whole system
five different ways. If the steps are hard-coded you will be editing your indexer
five times and comparing runs you cannot reproduce.

---

## Stage 3 — The two ways of searching

*Sessions 8 and 9. Lab 4 built both on six documents.*

### The incidence matrix

Build a term-document incidence matrix over a **500-document subset** — big
enough to be real, small enough to hold. One row per term, one column per
document, and a cell records presence, not a count, so the array should be
booleans.

Answer a query of the form `A AND B AND NOT C` with bitwise operations on whole
rows, the way session 8 did.

Then measure it, and report: how many cells, how many are non-zero, and what the
whole thing would cost over all 10,788 documents. **Read that last number before
you write your paragraph about it.** It does not say what session 8's argument
might lead you to expect at this scale. Make the claim your own measurements
support, about how the cost grows, rather than a claim about what it happens to
be here.

### The inverted index

Build an inverted index over all 10,788 documents: a dictionary from term to a
list of the documents containing it, sorted by document number.

**Store the count of the term in each document alongside the document number.**
Stage 4 needs it, and adding it afterwards means rebuilding.

Then implement query processing by **merging sorted lists**:

- `AND`, `OR` and `AND NOT`, each as a single walk down two sorted lists with
  one pointer in each.
- Multi-term `AND` with session 9's **rarest-first** optimisation: handle the
  terms in order of increasing document frequency, so the running result is as
  small as possible as early as possible.
- Evidence that rarest-first helps: time a three-term `AND` both ways and report
  both numbers.

Turning the postings lists into Python sets and intersecting them gives the same
answer and is genuinely fast. It is also not what is being marked: the merge is
what session 9 was about, and it is the merge that has to be in your code. Use
sets anywhere else you like.

A term that is not in the index makes a whole `AND` query empty. A query whose
words are all stop words leaves you with no terms at all; return nothing rather
than crashing.

---

## Stage 4 — Ranking

*Session 10. No lab did this one.*

Boolean retrieval returns a set. On this corpus, `crude AND oil AND opec`
returns 72 documents in no particular order, and nobody reads 72 documents. Add
ranking, exactly as session 10 defined it.

The weight of a term in a document is its count, compressed by a logarithm:

    weight = 1 + log10(count in this document)

The inverse document frequency of a term is the ratio of the collection size to
the number of documents containing it, also compressed:

    idf = log10(total documents / documents containing the term)

and a document's score for a query is the sum, over the query's terms, of the
first multiplied by the second:

    score = sum over query terms of (1 + log10(count)) × log10(N / df)

Return the highest-scoring documents first.

**Both logarithms are base 10**, as in the lesson. Base e changes every number in
your report and will not match any of the figures below.

Four requirements on how you compute it:

- **Score only the documents that can score above zero** — those on the postings
  list of at least one query term. Walking all 10,788 documents for every query
  produces the same ranking by the wrong algorithm, and it is roughly a hundred
  times slower.
- **Document frequency is the length of a postings list.** It needs no separate
  pass over anything. Say in your report what you cached and why.
- **Break ties by document number**, so that running the same query twice gives
  the same order and your report can quote a top ten that someone else can
  reproduce.
- **Keep a count of how many documents matched at all**, separately from the ten
  you return. Stage 6 has to display "10 results of 328", and the 328 is gone
  once you have taken the top ten. It is the size of the union of the query
  terms' postings lists and needs no scoring.

Two checks before you move on. A term appearing in *every* document must get an
idf of exactly zero and contribute nothing to any score — construct that case
and show it. And `opec` must outweigh `said` heavily.

---

## Stage 5 — Measuring your own choices

*The largest single block of marks, together with the tests.*

Every decision in stage 2 was handed to you as a choice, and so far you have only
been told that such choices matter. Here you find out, on your own system.

### The query set

These ten queries are fixed. Each is paired with the Reuters topic label that a
good answer ought to carry. Do not change them — your numbers are comparable
with anyone else's only because the query set is the same.

| query | relevant label |
|---|---|
| crude oil prices opec | `crude` |
| interest rate cut by the federal reserve | `interest` |
| wheat grain exports to the soviet union | `grain` |
| coffee quota talks | `coffee` |
| gold mining output | `gold` |
| japanese yen dollar currency intervention | `money-fx` |
| company fourth quarter earnings per share | `earn` |
| merger acquisition takeover bid | `acq` |
| sugar production quota | `sugar` |
| shipping freight rates | `ship` |

For one query, take your top ten documents and count how many carry the paired
label. That fraction is **precision at 10**. Average it over the ten queries and
you have a single number describing your system.

It is a rough measure and you should treat it as one in your report. A document
about the oil price that Reuters happened not to label `crude` counts against
you unfairly, and ten queries is a small sample. It is nevertheless far better
than looking at the results and deciding they seem reasonable, and it has the
one property that matters: it is computed the same way before and after a
change, so a difference between two runs means something.

### The comparison

Run your whole system under each of these five settings, and record for each:
vocabulary size, total postings, how long the index took to build, mean query
time, and mean precision at 10.

| # | setting |
|---|---|
| 1 | case folding only |
| 2 | plus stop word removal |
| 3 | plus Porter stemming |
| 4 | plus lemmatization instead of stemming |
| 5 | stemming, with the stop list taken back out |

Build the index once per setting and change **nothing** between runs except the
pipeline settings.

Then write the paragraph the table exists for. Which step changed the results
most? Did any step make them **worse**? Session 4 warned that removing stop words
is sometimes a mistake and session 5 that stemming and lemmatization are not
interchangeable. You now have evidence about this corpus, and evidence about one
corpus is exactly what you should claim — no more.

### What idf does to stop words

Session 10 claimed that TF-IDF suppresses common words without needing a stop
list at all, because a word in every document has an idf of zero.

Test that claim with **setting 5 against setting 3**. Those two differ in exactly
one thing — one has the stop list, the other leaves the stop words in the index
and lets idf deal with them — so a difference between them is attributable to
that one thing. Comparing against setting 1 would change stemming at the same
time and tell you nothing about the claim.

While you are there, sort your vocabulary by document frequency and read the top
thirty terms. NLTK's stop list was written for English in general, not for
newswire, and this corpus contains words that behave exactly like stop words
without appearing on any list. **Name one in your report**, give its document
frequency and its idf, and say what a stop list would have had to know about this
corpus in order to contain it.

### Subword tokenization

*Session 3.*

Every step so far has assumed the word is the unit. Train a BPE tokenizer on this
corpus with a vocabulary of 8,000 — Hugging Face's `tokenizers` library does this
in a few lines — and report three things:

1. How many distinct word types your pipeline needs to cover the corpus, against
   BPE's 8,000.
2. What BPE does with a word your word index has never seen. Take a word that
   does not occur in Reuters and encode it.
3. What the pieces look like. `unbelievably` comes back as `un`, `belie`, `v`,
   `ably`, which are **not** the morphemes session 3 described. Explain briefly
   why not.

You are **not** asked to build a second index on subword units. Instead, write
one paragraph on why subword tokens are awkward as index terms for a search
engine, given what stage 4 does with document frequency.

---

## Stage 6 — The interface

Put a Streamlit page in front of your index. One page, with:

- **A search box**, and a choice between Boolean and ranked retrieval.
- **Results**: headline, score, topic labels, and a snippet of the body with the
  query's words marked. The marking has to survive stemming — someone searching
  for `mining` should see `mines` marked if that is why the document matched,
  which means you cannot simply look for the typed string in the text. Say in
  your report how you did it.
- **A line reading** "10 results of 328 in 24 ms".
- **A way to switch between the stage 5 settings**, so a visitor can watch the
  ablation happen instead of reading the table.

That last one has a trap in it, and noticing the trap is part of the exercise.
You cannot change the pipeline settings and search the same index: the terms in
that index were produced by those settings, so an unstemmed query against a
stemmed index is precisely the mismatch stage 2 was about. Each setting needs its
own index, built ahead of time.

Two practical points. **Build the indexes ahead of time and save them**; the page
should load them, never build them, and should start in about a second rather
than the minute the build takes. And **load them once**, not on every keystroke —
Streamlit re-runs your script top to bottom every time anything on the page
changes, and it gives you a caching decorator for exactly this.

The interface is worth 10 marks of 100. It should be clean. It does not need to
be beautiful.

---

## Numbers your system must reproduce

These are checks on whether your engine is right. If yours disagree, something is
wrong upstream, and the note beside each one says where to look first.

All of them assume the full pipeline — case folding, stop word removal, Porter
stemming — over all 10,788 documents in sorted file order.

| what | value | if yours differs |
|---|---|---|
| documents in the collection | 10,788 | you are not reading the whole corpus |
| vocabulary size | 20,447 | your stage 2 differs — check the alphabetic-only rule and the stemmer |
| total postings | 508,902 | as above |
| documents matching `crude AND oil AND opec` | 72 | your merge or your stemming |
| idf of `opec` (document frequency 121) | 1.950 | base of the logarithm, or the document count |
| idf of `oil` (document frequency 1,065) | 1.006 | as above |
| idf of `said` (document frequency 6,784) | 0.201 | as above |
| idf of `gold` (document frequency 225) | 1.681 | as above |
| documents matching `crude oil prices opec` at all | 2,275 | your union count for the "of 328" line |
| mean precision at 10 over the ten queries | about 0.90 | see below |
| incidence matrix over the first 500 documents | 4,356 terms, 2,178,000 cells, 24,788 non-zero (1.14%) | your stage 2, or your subset is not the first 500 |
| the same matrix over all 10,788 documents | 220,582,236 cells, 28 MB packed one bit per cell | arithmetic only |

**On that precision figure.** Anything from about 0.80 upwards is a working
system. Below roughly 0.6, something is wrong, and the usual cause is the one
this project is built around: the query being normalised differently from the
index, so `OPEC` never meets `opec`. Check that before you check anything else.

A mean of **1.00 is not a triumph**, it is a bug — almost certainly your ranking
is reading the Reuters labels somewhere, which it must not.

The five-row ablation table is deliberately **not** here. Producing it is the
work of stage 5, and reading it is what carries the marks.

### A hand-check for your merges

Before you index 10,788 documents, index these six, with case folding on and
both stop-word removal and stemming **off**. They are lab 4's collection:

1. The cat sat on the mat.
2. The dog sat on the log.
3. The cat and the dog played together.
4. Dogs and cats are good friends.
5. A bird sang in the tall tree.
6. The cat chased the bird up the tree.

You should get a vocabulary of **23** terms; `cat` in documents 1, 3 and 6;
`dog` in 2 and 3; `bird` in 5 and 6; `cat AND bird` giving document 6; and
`cat AND NOT dog` giving 1 and 6. Those are the numbers lab 4 printed, so you can
check them against your own notes.

Now turn stemming on and run it again. `cat` picks up document 4, because `cats`
and `cat` are now one term. That change is the whole of session 5 in one line,
and it is worth seeing on six documents before you meet it on eleven thousand.

---

## Testing

**You must submit a test suite, and it is worth 10 marks.**

This is not a request for coverage figures. It is a request for tests that pin
down the things you claim. At a minimum:

1. **A real failure of your stage 1 extractor.** Your report describes one; this
   test proves you found it rather than assumed it.
2. **A word where stemming and lemmatization genuinely differ**, with what each
   produces asserted.
3. **The corpus-specific stop word you found**, with its document frequency and
   idf. Then the number in the report and the number in the code cannot drift
   apart.
4. **A query your system handles badly**, with the behaviour pinned down. A
   failing search held in place by a passing test is worth more here than
   pretending the failure is not there.
5. **A test you wrote because something broke.** Over three weeks you will fix at
   least one real bug. When you do, write the test that would have caught it,
   before you fix it.

Beyond those, test your merges on short lists where you know the answers by hand,
and test the arithmetic of stage 4 on a handful of made-up documents rather than
on Reuters. Tests that take a minute to run are tests you stop running.

---

## The report

At most four pages, including tables. It is not a description of your code. It
contains:

- the five-row ablation table, and the paragraph that reads it
- your answer on the incidence matrix and how its cost grows
- the corpus-specific stop word you found, with its numbers
- the BPE comparison
- the rarest-first timings, both ways
- one honest paragraph on **what your system does badly**, with a query that
  shows it

Every system has one. Finding your own is worth more than not having one.

**Every number in the report must come out of code you are submitting.** A number
you cannot reproduce on demand loses the mark it was supporting.

---

## What to hand in

One repository, as a link or a zip, containing:

- your engine, importable and with no Streamlit imports in it
- the Streamlit page
- whatever builds and saves the indexes
- something that reproduces the ablation table on demand
- your tests
- `report.md`
- `README.md` — how to run it, from a clone to a search box, and a section
  naming where you used an AI assistant
- your pinned dependency list

You will also give a **three-minute demo**: run two queries, one that works and
one that does not, and answer questions about your ablation table.

---

## How it is marked

| | marks | |
|---|---|---|
| Stage 1 — regex extraction | 5 | Fields extracted; one failure found and explained. |
| Stage 2 — the pipeline | 15 | One switchable pipeline, used for both indexing and queries. |
| Stage 3 — index and Boolean | 20 | Both structures; merges rather than sets; rarest-first timed. |
| Stage 4 — TF-IDF ranking | 20 | Correct weights; only candidate documents scored. |
| Stage 5 — measurement and report | 20 | The table, and the reading of it. |
| Stage 6 — interface | 10 | Works, loads rather than builds, marks the right words. |
| Your tests | 10 | Whether they test what your report actually claims. |

Marks are lost for a system that cannot be run from your README, for numbers in
the report that your own code does not reproduce, and for claims stated without
the evidence to support them.

**A precision of 0.62 that you understand and can explain scores above a 0.90
that you cannot.**

---

## Rules

- **Individual submission.** Discuss the ideas with anyone. The code and the
  report are yours.
- **AI assistants are permitted**, and you must say where you used one, in your
  README. The demo is where this is checked: you will be asked to explain any
  part of your code.
- **Deadline:** *[date]*. Demos in the lab session of *[date]*.

## What is not in this project

Sessions 6 and 7 — part-of-speech tagging and named entity recognition — are
deliberately excluded, and so is everything from Unit II onwards: no word
embeddings, no classifier, no neural network.

In particular, **cosine similarity is not part of this project.** It arrives in
session 19. Ranking here is the sum of TF-IDF weights that session 10 defined.
Using cosine early is not rewarded, and describing your ranking as cosine when it
is a sum costs marks.

## Where each stage comes from

| stage | sessions | lab |
|---|---|---|
| 1 · regex extraction | 1, 2 | Lab 2 |
| 2 · the pipeline | 3, 4, 5 | Lab 3 |
| 3 · index and Boolean | 8, 9 | Lab 4 |
| 4 · TF-IDF ranking | 10 | — |
| 5 · measurement | 3, 4, 5, 10 | — |
| 6 · interface | — | — |

Stages 4 and 5 have no lab behind them. If you are stuck, the session is the
place to go back to before the internet is.
