#!/usr/bin/env python3
"""Fetch the model and corpus files that pip does not install.

    uv run bootstrap.py

spaCy's English model and NLTK's corpora are downloaded at runtime, not
resolved as dependencies. A demo that dies on

    OSError: Can't find model 'en_core_web_sm'

halfway through a lecture is exactly what this workspace exists to prevent.
Run this once after cloning. It is safe to run again — anything already
present is skipped.
"""

import subprocess
import sys

NLTK_PACKAGES = [
    "punkt",                          # sentence splitter
    "punkt_tab",                      # its tables, needed since nltk 3.8.2
    "stopwords",                      # session 04
    "wordnet",                        # lemmatizer, session 05
    "words",                          # english word list
    "averaged_perceptron_tagger_eng",  # POS tagger, session 06
    "gutenberg",                      # a real corpus to count words in
]

SPACY_MODEL = "en_core_web_sm"


def fetch_spacy() -> bool:
    """Download the small English pipeline unless it already loads."""
    import spacy

    try:
        spacy.load(SPACY_MODEL)
    except OSError:
        pass
    else:
        print(f"spacy  {SPACY_MODEL:34} already present")
        return True

    print(f"spacy  {SPACY_MODEL:34} downloading...")
    result = subprocess.run([sys.executable, "-m", "spacy", "download", SPACY_MODEL])
    ok = result.returncode == 0
    print(f"spacy  {SPACY_MODEL:34} {'ok' if ok else 'FAILED'}")
    return ok


def fetch_nltk() -> bool:
    """Download each NLTK package. Returns False if any one failed."""
    import nltk

    every_one_ok = True
    for pkg in NLTK_PACKAGES:
        ok = nltk.download(pkg, quiet=True)
        every_one_ok = every_one_ok and ok
        print(f"nltk   {pkg:34} {'ok' if ok else 'FAILED'}")
    return every_one_ok


def main() -> int:
    print("Fetching models and corpora. First run takes a minute or two.\n")
    nltk_ok = fetch_nltk()
    spacy_ok = fetch_spacy()

    if nltk_ok and spacy_ok:
        print("\nAll set. Try: uv run demos/s04_unicode.py")
        return 0

    print("\nSomething did not download. Check your network and run this again.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
