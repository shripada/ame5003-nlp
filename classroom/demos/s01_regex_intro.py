#!/usr/bin/env python3
"""Session 01 — regular expressions from scratch, and why `re` is not enough.

    uv run demos/s01_regex_intro.py

Same examples as lessons/0001-what-is-nlp-and-your-first-tool.html, so what
runs here is what they read.

Sections 1-7 use the standard library `re`, because that is what you reach for
first. Section 8 breaks it on Kannada. Section 9 introduces `regex`, which is
what every later demo in this course imports.
"""

import re
import unicodedata

import regex


def head(title: str) -> None:
    print(f"\n{title}\n{'-' * len(title)}")


WIDTH = 19  # widest pattern below, so results line up in one column


def show(pattern: str, text: str, note: str = "", module=re) -> None:
    """Run findall and print the pattern, the result, and an optional note."""
    print(f"  {pattern:<{WIDTH}} {module.findall(pattern, text)}")
    if note:
        print(f"  {'':<{WIDTH}} ^ {note}")


# ── 1. literals ──────────────────────────────────────────────────────────
head("1. A pattern is mostly just the text you are looking for")

text = "The cat sat. A Cat stood."
print(f"text  {text!r}\n")

show(r"cat", text, "plain letters match themselves — and only themselves")
show(r"[Cc]at", text, "brackets = 'any ONE of these', so both cases match")


# ── 2. character classes ─────────────────────────────────────────────────
head("2. Character classes: [...] is one character from a set")

text = "Room 7, floor 12, block B"
print(f"text  {text!r}\n")

show(r"[0-9]", text, "a range: one digit")
show(r"[0-9]+", text, "'+' = one or more, so digits stay glued together")
show(r"[^0-9 ]+", text, "'^' FIRST INSIDE brackets flips it: NOT these")

print("""
  Careful — '^' means two different things:
    [^abc]   inside brackets   NOT a, b or c
    ^The     outside brackets  start of the string""")
print()
print(f"  {'^Room':<{WIDTH}} {bool(re.search(r'^Room', text))}    'Room' does start the text")
print(f"  {'^floor':<{WIDTH}} {bool(re.search(r'^floor', text))}   'floor' is in the middle, so the anchor fails")


# ── 3. shorthands ────────────────────────────────────────────────────────
head("3. Shorthands for the classes you need constantly")

text = "Roll no 7 is A_1."
print(f"text  {text!r}\n")

show(r"\d", text, r"\d  a digit, same as [0-9]")
show(r"\w+", text, r"\w  a 'word' character: letter, digit or underscore")
show(r"\s", text, r"\s  whitespace — here, the four spaces")
show(r"\S+", text, r"uppercase inverts: \S is NOT whitespace, so it keeps the '.'")


# ── 4. quantifiers ───────────────────────────────────────────────────────
head("4. Quantifiers: how MANY of the thing before it")

print("""\
  *      zero or more      a*      '', 'a', 'aaa'
  +      one or more       a+      'a', 'aaa'  (not '')
  ?      zero or one       a?      '', 'a'
  {3}    exactly three     a{3}    'aaa'
  {2,4}  two to four       a{2,4}  'aa', 'aaa', 'aaaa'""")

spellings = "color colour colr colooor coloXr"
print(f"\ntext  {spellings!r}\n")

show(r"colo*r", spellings, "'*' applies to 'o' alone: colr and colooor sneak in")
show(r"colo+r", spellings, "'+' needs at least one 'o', so colr is out")
show(r"colo.r", spellings, "'.' is ANY character — it happily matches the X")
show(r"colou?r", spellings, "the 'u' is optional — this is the one you wanted")


# ── 5. the dot, and escaping ─────────────────────────────────────────────
head("5. '.' matches anything, so escape it when you mean a real dot")

text = "file.txt fileAtxt v1.2"
print(f"text  {text!r}\n")

show(r"file.txt", text, "'.' matched the literal A too — probably not intended")
show(r"file\.txt", text, r"'\.' means an actual full stop")
print()
print(r"  Same for any character with a job: \+ \* \? \( \) \[ \] \\ \$ \^")


# ── 6. the functions you will actually call ──────────────────────────────
head("6. findall is not the only one")

text = "Due 15/03/2026, exam 01/04/2026"
date = r"\d{2}/\d{2}/\d{4}"
print(f"text     {text!r}")
print(f"pattern  {date}\n")

print(f"  findall  {re.findall(date, text)}")
print("            ^ every match, as a list")
print(f"  search   {re.search(date, text)}")
print("            ^ a Match object for the FIRST hit, or None")
print(f"  .group() {re.search(date, text).group()!r}   the matched text")
print(f"  .span()  {re.search(date, text).span()}      where it sat")
print(f"  match    {re.match(date, text)}")
print("            ^ None: match() only looks at the START of the string")
print(f"  sub      {re.sub(date, '<DATE>', text)!r}")
print(f"  split    {re.split(r',\s*', text)}")


# ── 7. a real example, and being specific ────────────────────────────────
head("7. Pulling real fields out of real text")

text = "Call +91 98765 43210 before 15/03/2026 or email admin@manipal.edu"
print(f"text  {text!r}\n")

show(r"\d{2}/\d{2}/\d{4}", text, "date")
show(r"\+91 \d{5} \d{5}", text, r"phone — note the escaped \+")
show(r"[\w.]+@[\w.]+", text, "email — crude, but enough for today")

print()
print("  Be specific. A loose pattern matches things you never meant:")

rolls = "MSIS2026001 msis2026001 12345678901"
print(f"\ntext  {rolls!r}\n")

show(r"[A-Z]{4}\d{7}", rolls, "four capitals then seven digits — only the real one")
show(r"\w{4}\d{7}", rolls, r"\w covers digits and lowercase too, so junk gets in")


# ── 8. where re gives up ─────────────────────────────────────────────────
head("8. The problem: run `re` on Kannada")

text = "ಕನ್ನಡ ಭಾಷೆ"
print(f'text  {text!r}   "Kannada language" — two words\n')

found = re.findall(r"\w+", text)
print(rf"  re.findall(r'\w+', text)     {found}")
print(f"                               ^ {len(found)} pieces, not 2")

print("\n  Why? Look at what the first word is made of:\n")
for ch in "ಕನ್ನಡ":
    is_letter = unicodedata.category(ch).startswith("L")
    kind = "a letter" if is_letter else "a combining mark  <--"
    print(f"    {ch}   U+{ord(ch):04X}   {unicodedata.name(ch):<24} {kind}")

print()
print(r"  The stdlib's \w counts letters and digits. A combining mark is")
print(r"  neither, so \w treats the virama as a boundary and cuts the word")
print("  in half. Every Indic script, and plenty of others, break this way.")


# ── 9. the fix: the regex module ─────────────────────────────────────────
head("9. The fix: `regex`, a drop-in replacement that knows Unicode")

print("  pip install regex     (or, here: uv add regex)\n")
print(rf"  regex.findall(r'\w+', text)  {regex.findall(r'\w+', text)}")
print("                               ^ two words, as a human would count them")

print("\n  It also adds Unicode property classes, written \\p{...}:\n")

word = "ಕನ್ನಡ"
print(f"  text  {word!r}\n")
show(r"\p{L}+", word, r"\p{L} = any Letter — still splits, marks are not letters",
     module=regex)
show(r"[\p{L}\p{M}]+", word, r"\p{M} = any Mark. Letters AND marks: correct",
     module=regex)
show(r"\w+", word, r"regex's own \w already includes the marks", module=regex)

print("\n  Properties worth knowing:")
print(r"    \p{L}   any letter, in any script      \p{Lu}  uppercase letter")
print(r"    \p{M}   combining mark                 \p{N}   any number")
print(r"    \p{P}   punctuation                    \p{Script=Kannada}")

script = "Kannada ಕನ್ನಡ 2026!"
print(f"\n  text  {script!r}\n")
show(r"\p{Script=Kannada}+", script, "only the Kannada run", module=regex)
show(r"\p{Lu}", script, "only uppercase letters", module=regex)
show(r"\p{N}+", script, "any number", module=regex)
show(r"\p{P}", script, "punctuation", module=regex)


# ── what to remember ─────────────────────────────────────────────────────
head("What to remember")

print("""\
  1. A pattern is literals plus classes plus quantifiers. That is most of it.
  2. '^' means 'not' inside [], and 'start of string' outside it.""")
print(r"  3. Escape anything with a job: \. \+ \? \( \)")
print("  4. Be specific — a loose pattern quietly matches the wrong thing.")
print(r"  5. The stdlib's \w assumes English. It splits Kannada mid-word.")
print("  6. So: import regex, not re. Every later demo in this course does.")
print()
