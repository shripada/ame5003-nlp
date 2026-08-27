#!/usr/bin/env python3
"""Session 04 — Unicode: why two identical-looking strings are not equal.

    uv run demos/s04_unicode.py

Same café example as lessons/0004-normalization-and-stop-words.html.

The short version: a character you see on screen is not always one character
in memory. Compare before you normalize and you get wrong answers that are
very hard to spot, because the screen shows you nothing is wrong.
"""

import unicodedata


def head(title: str) -> None:
    print(f"\n{title}\n{'-' * len(title)}")


# ── two cafés ────────────────────────────────────────────────────────────
head("1. Two strings that look the same and are not")

a = "café"        # é as ONE character
b = "café"       # e + a separate combining accent

print(f"  a = {a}      len {len(a)}")
print(f"  b = {b}      len {len(b)}")
print()
print(f"  a == b   ->   {a == b}")
print()
print("  what they are really made of:")
for name, s in (("a", a), ("b", b)):
    print(f"    {name}:")
    for c in s:
        print(f"       U+{ord(c):04X}  {unicodedata.name(c, '?')}")


# ── normalize, then compare ──────────────────────────────────────────────
head("2. Normalize first, then compare")
# NFC: Normalization Form Canonical Composition. This is the default normalization form
print("""\
  NFC composes: e + accent  ->  the single é
  NFD decomposes: the single é  ->  e + accent""")
print()
print(f"  NFC(a) == NFC(b)   ->   {unicodedata.normalize('NFC', a) == unicodedata.normalize('NFC', b)}")
print(f"  NFD(a) == NFD(b)   ->   {unicodedata.normalize('NFD', a) == unicodedata.normalize('NFD', b)}")
print()
print(f"  len NFC  a={len(unicodedata.normalize('NFC', a))}  b={len(unicodedata.normalize('NFC', b))}")
print(f"  len NFD  a={len(unicodedata.normalize('NFD', a))}  b={len(unicodedata.normalize('NFD', b))}")
print("""
  Pick one form and normalize everything on the way in. NFC is the
  usual choice — it is what the web is mostly already in.""")


# ── lower() is not enough ────────────────────────────────────────────────
head("3. lower() is not casefold()")

for word in ["STRASSE", "Straße", "İstanbul"]:
    print(f"  {word:10}  lower {word.lower():12}  casefold {word.casefold()}")

print()
print("  German ß uppercases to SS. Only casefold() knows that, so")
print(f"  'Straße'.lower() == 'strasse'      ->  {'Straße'.lower() == 'strasse'}")
print(f"  'Straße'.casefold() == 'strasse'   ->  {'Straße'.casefold() == 'strasse'}")
print("  Use casefold() when you are comparing, lower() when you are printing.")


# ── the one that breaks file reading ─────────────────────────────────────
head("4. Bytes are not text")

text = "café costs €3"
print(f"  {text!r}")
print()
for encoding in ("utf-8", "utf-16", "latin-1"):
    try:
        raw = text.encode(encoding)
        print(f"  {encoding:9} {len(raw):>3} bytes   {raw[:20]!r}")
    except UnicodeEncodeError as e:
        print(f"  {encoding:9} FAILS — {e.reason}: {e.object[e.start:e.end]!r}")

print("""
  Same text, different byte counts. This is why open() needs
  encoding='utf-8' — guessing wrong gives you mojibake or a crash.""")

wrong = text.encode("utf-8").decode("latin-1")
print(f"\n  utf-8 bytes read as latin-1:  {wrong!r}")
print("  Nothing raised an error. It just quietly became wrong.")

print()
