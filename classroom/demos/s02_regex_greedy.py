#!/usr/bin/env python3
"""Session 02 — groups, non-capturing groups, and the greedy trap.

    uv run demos/s02_regex_greedy.py

Same examples as lessons/0002-regex-groups-and-the-greedy-trap.html, so what
runs here is what they read.
"""

import regex


def head(title: str) -> None:
    print(f"\n{title}\n{'-' * len(title)}")


# ── groups pull pieces out of a match ────────────────────────────────────
head("1. Groups: parentheses name the pieces you want back")

text = "Exam on 15/03/2026."
pattern = r"(\d{2})/(\d{2})/(\d{4})"

m = regex.search(pattern, text)
print(f"text     {text!r}")
print(f"pattern  {pattern}")
print()
print(f"  group(0)  {m.group(0)!r}   the whole match")
print(f"  group(1)  {m.group(1)!r}           the day")
print(f"  group(2)  {m.group(2)!r}           the month")
print(f"  group(3)  {m.group(3)!r}         the year")


# ── the findall surprise ─────────────────────────────────────────────────
head("2. findall returns tuples the moment you add a group")

text = "Due 15/03/2026 and 01/04/2026"
print(f"text  {text!r}\n")

print("  with groups     ", regex.findall(r"(\d{2})/(\d{2})/(\d{4})", text))
print("                   ^ tuples of pieces, not the dates")
print()
print("  no groups       ", regex.findall(r"\d{2}/\d{2}/\d{4}", text))
print("  (?:...) groups  ", regex.findall(r"(?:\d{2})/(?:\d{2})/\d{4}", text))
print("                   ^ (?:...) groups for structure without capturing")


# ── the greedy trap ──────────────────────────────────────────────────────
head("3. The greedy trap: + takes as much as it can")

html = "<b>bold</b> and <i>italic</i>"
print(f"text  {html!r}\n")

print("  <.+>   ", regex.findall(r"<.+>", html))
print("          ^ one giant match: it ran to the LAST '>' in the line")
print()
print("  <.+?>  ", regex.findall(r"<.+?>", html))
print("          ^ '?' makes it lazy: stop at the first '>' that works")
print()
print("  <[^>]+>", regex.findall(r"<[^>]+>", html))
print("          ^ better still: say what you mean — 'not a >'")


# ── alternation ──────────────────────────────────────────────────────────
head("4. Alternation, and why the group matters")

text = "one cats, two dog"
print(f"text  {text!r}\n")
print("  (?:cat|dog)s?  ", regex.findall(r"(?:cat|dog)s?", text))
print("                  ^ the s? applies to cat OR dog, as intended")
print()
print("  cat|dogs?      ", regex.findall(r"cat|dogs?", text))
print("                  ^ no group: | splits the WHOLE pattern, so this reads")
print("                    as 'cat' OR 'dogs?' — the plural on cats is lost")

print()
