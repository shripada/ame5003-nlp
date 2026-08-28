#!/usr/bin/env python3
"""Session 07 — named entity recognition, and the BIO tags underneath it.

    uv run demos/s07_ner.py

Same sentences as lessons/0007-named-entity-recognition.html, run through spaCy
(en_core_web_sm). The failure on "Rao paid Infosys 5000 rupees last week" is
real output, not a staged example, and so is the merged "Google Microsoft".

The short version: an entity is a span, not a word, so a per-word tagger needs
the BIO scheme to mark where a name starts and stops.
"""

import spacy


def head(title: str) -> None:
    print(f"\n{title}\n{'-' * len(title)}")


# The same pipeline as session 6. The tagger we used there and the entity
# recogniser used here are two components of it, run one after the other on
# the same Doc, which is why every token below carries both a .pos_ and an
# entity tag.
nlp = spacy.load("en_core_web_sm")


def show_entities(sentence: str) -> None:
    """Print each entity spaCy found, with its type spelled out."""
    doc = nlp(sentence)
    print(f"    {sentence}")
    if not doc.ents:
        print("      (no entities found)")
        return
    for ent in doc.ents:
        # spacy.explain turns a label such as GPE into the English phrase it
        # stands for, which saves memorising the list.
        print(f"      {ent.text:26} -> {ent.label_:8} ({spacy.explain(ent.label_)})")


# ── what the task returns ────────────────────────────────────────────────
head("1. The names in a sentence, with their types")

show_entities("The Reserve Bank of India raised rates in Mumbai on Monday.")
print("""
  Session 6 labelled one word at a time, which worked because a word's
  grammatical class belongs to that word alone. 'The Reserve Bank of
  India' is one organization spread over five words. The name is the
  span, not any word in it, and that is the new difficulty.""")


# ── BIO ──────────────────────────────────────────────────────────────────
head("2. BIO: how one tag per word marks a span")

doc = nlp("The Gang of India raised rates.")
for token in doc:
    # ent_iob_ is the B / I / O part and ent_type_ the label. spaCy keeps them
    # in separate attributes; joined with a hyphen they are the notation the
    # lesson prints on the board.
    tag = f"{token.ent_iob_}-{token.ent_type_}" if token.ent_type_ else "O"
    print(f"    {token.text:10} {tag}")

print("""
  B- begins an entity, I- continues the one already begun, O is outside
  any entity. One B-ORG followed by four I-ORG tags reconstructs the
  whole span. The machinery is session 6's, unchanged — the scheme is
  what lets it mark spans.""")


# ── why B- has to exist ──────────────────────────────────────────────────
head("3. Why a separate B- tag is needed")

print("""\
  Two names of the same type can sit next to each other, and the boundary
  between them has to survive the tagging. Watch what happens when they do:""")
print()
for token in nlp("Google Microsoft and Meta all reported."):
    tag = f"{token.ent_iob_}-{token.ent_type_}" if token.ent_type_ else "O"
    print(f"    {token.text:10} {tag}")

print("""
  spaCy read 'Google Microsoft' as a single two-word company: B-ORG then
  I-ORG. That is the wrong reading of the sentence, and it is also a
  demonstration of what the tags mean. Had it seen two companies it would
  have written B-ORG twice, as it did for 'Meta'. Tagging every word
  simply ORG would lose the distinction entirely, which is the reason the
  B- tag exists.""")


# ── ambiguity ────────────────────────────────────────────────────────────
head("4. The same name, more than one type")

for sentence in ["Washington signed the treaty in 1795.",
                 "She flew to Washington for the summit.",
                 "Apple sued Samsung over the patent.",
                 "She ate an apple from the tree."]:
    doc = nlp(sentence)
    # Pull out whichever entity covers the ambiguous word, if any was found.
    named = [(e.text, e.label_) for e in doc.ents
             if e.text.lower() in {"washington", "apple"}]
    print(f"    {sentence:42} {named if named else 'no entity here'}")

print("""
  'Apple' is an ORG in one sentence and not an entity at all in the other,
  which is the ambiguity resolved correctly from context. 'Washington' is
  not: this model calls it a GPE in both, including the sentence where it
  is plainly a person. The ambiguity is real; a small model does not
  always resolve it.""")


# ── the lab 2 debt ───────────────────────────────────────────────────────
head("5. The merchant name lab 2 could not extract")

print("""\
  Lab 2 pulled dates and rupee amounts out of bank messages with regular
  expressions and got them reliably, because those have a fixed shape. The
  merchant name defeated it, because names have no shape to match. This is
  the debt being paid — the same task, done by a learned model:""")
print()
show_entities("Rs 450 was debited for Swiggy on 12 August.")
print("""
  Swiggy comes out as an ORG with no pattern written for it. Now the same
  message in the capitals a bank actually sends:""")
print()
show_entities("INR 450.00 debited from your account for SWIGGY on 12-08-2026.")
print("""
  The merchant is gone, and 'INR' is labelled a company. Capitalisation is
  one of the strongest clues the model learned, and SHOUTED TEXT destroys
  it. Note what this does to session 4: lowercasing everything, which was
  the right first move for search, would take this model's best clue away.""")


# ── the honest failure ───────────────────────────────────────────────────
head("6. A sentence the recogniser gets wrong")

show_entities("Rao paid Infosys 5000 rupees last week.")
print("""
  'Rao' and 'last week' are right. 'Infosys 5000' is one span labelled
  PRODUCT, so the company and the amount have been glued together and the
  result mislabelled, and '5000 rupees' is missed as MONEY entirely.

  The parts that resemble the training data came out right and the parts
  that do not came out wrong. A model trained largely on Western news text
  is weakest on exactly the names an Indian application cares about. This
  is a limit to measure before trusting the output, not a reason to avoid
  NER — fine-tuning on Indian data is the fix, and that is lab 12.""")

print()
