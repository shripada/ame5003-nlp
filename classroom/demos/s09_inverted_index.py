#!/usr/bin/env python3
"""Session 09 — the inverted index, and the two-pointer merge.

    uv run demos/s09_inverted_index.py

The same six plays and the same query as session 8, so the answer had better
come out the same: lessons/0009-the-inverted-index.html. The merge trace
printed here is produced by the merge itself, not typed out beside it.

The short version: keep only the ones. For each term, a sorted list of the
documents it appears in, intersected by walking two lists at once.
"""


def head(title: str) -> None:
    print(f"\n{title}\n{'-' * len(title)}")


PLAYS = ["Antony and Cleopatra", "Julius Caesar", "The Tempest",
         "Hamlet", "Othello", "Macbeth"]

# Session 8's matrix, unchanged, so the two demos can be run back to back and
# compared. Everything below is derived from it.
MATRIX = {
    "Antony":    [1, 1, 0, 0, 0, 1],
    "Brutus":    [1, 1, 0, 1, 0, 0],
    "Caesar":    [1, 1, 0, 1, 1, 1],
    "Calpurnia": [0, 1, 0, 0, 0, 0],
    "Cleopatra": [1, 0, 0, 0, 0, 0],
    "mercy":     [1, 0, 1, 1, 1, 1],
    "worser":    [1, 0, 1, 1, 1, 0],
}


def names(doc_ids: list[int]) -> str:
    return ", ".join(PLAYS[d - 1] for d in doc_ids)


# ── building it ──────────────────────────────────────────────────────────
head("1. Inverting the matrix")

# Keep the 1s and throw the 0s away. enumerate from 1 because the plays are
# numbered 1 to 6, as they were on the board; sorted order comes free from
# reading the row left to right, and everything below depends on it.
index = {
    term: [doc_id for doc_id, bit in enumerate(bits, start=1) if bit]
    for term, bits in MATRIX.items()
}

width = max(len(t) for t in index) + 2
for term, postings in index.items():
    print(f"    {term.ljust(width)}{' -> '.join(str(d) for d in postings)}")

print("""
  A dictionary of terms, and for each one a postings list: the IDs of the
  documents containing it, in sorted order. It is called inverted because
  a document normally points at its words; here a word points at its
  documents.""")
print()
print(f"    matrix row for Caesar    {' '.join(str(b) for b in MATRIX['Caesar'])}"
      f"    ({len(MATRIX['Caesar'])} cells)")
print(f"    postings list            {' '.join(str(d) for d in index['Caesar'])}"
      f"      ({len(index['Caesar'])} numbers, no zeros)")


# ── the merge ────────────────────────────────────────────────────────────
head("2. Intersecting two postings lists")

def intersect(p1: list[int], p2: list[int], trace: bool = False) -> list[int]:
    """The documents in both lists, by walking each list once.

    Both lists are sorted, so when one ID is smaller than the other it cannot
    appear later in the other list — every remaining entry there is larger —
    and it can be skipped without ever being reconsidered. That is what lets
    each pointer move only forward.
    """
    result: list[int] = []
    i = j = 0
    step = 0
    while i < len(p1) and j < len(p2):
        step += 1
        if p1[i] == p2[j]:
            if trace:
                print(f"    step {step}:  {p1[i]} vs {p2[j]}   equal   "
                      f"-> keep {p1[i]}, advance both")
            result.append(p1[i])
            i += 1
            j += 1
        elif p1[i] < p2[j]:
            if trace:
                print(f"    step {step}:  {p1[i]} vs {p2[j]}   {p1[i]} < {p2[j]}   "
                      f"-> advance the first list")
            i += 1
        else:
            if trace:
                print(f"    step {step}:  {p1[i]} vs {p2[j]}   {p1[i]} > {p2[j]}   "
                      f"-> advance the second list")
            j += 1
    if trace:
        exhausted = "first" if i == len(p1) else "second"
        print(f"             the {exhausted} list is finished -> stop")
    return result


print("    Brutus AND mercy")
print()
print(f"    Brutus:  {' -> '.join(str(d) for d in index['Brutus'])}")
print(f"    mercy:   {' -> '.join(str(d) for d in index['mercy'])}")
print()
merged = intersect(index["Brutus"], index["mercy"], trace=True)
print()
print(f"    result:  {' -> '.join(str(d) for d in merged)}      ({names(merged)})")

x, y = len(index["Brutus"]), len(index["mercy"])
print(f"""
  Each pointer moves forward only, so each list is read once and no entry
  is ever revisited. Lists of lengths {x} and {y} cost at most {x + y} steps between
  them, against the {x * y} comparisons that checking every pair would need:
  about x + y work instead of x × y.
  This is possible only because the lists are sorted, which is why they are
  kept sorted when the index is built.""")


# ── the full query ───────────────────────────────────────────────────────
head("3. The whole query from session 8")

def and_not(postings: list[int], exclude: list[int]) -> list[int]:
    """Everything in the first list that is not in the second."""
    removed = set(exclude)
    return [d for d in postings if d not in removed]


print("    Brutus AND Caesar AND NOT Calpurnia")
print()
step1 = intersect(index["Brutus"], index["Caesar"])
step2 = and_not(step1, index["Calpurnia"])
print(f"    Brutus  {' '.join(map(str, index['Brutus']))}   INTERSECT   "
      f"Caesar  {' '.join(map(str, index['Caesar']))}     ->   {' '.join(map(str, step1))}")
print(f"            {' '.join(map(str, step1))}   AND NOT     "
      f"Calpurnia  {' '.join(map(str, index['Calpurnia']))}           ->   "
      f"{' '.join(map(str, step2))}")
print()
print(f"    -> {names(step2)}")
print("""
  Session 8 got the same two plays out of the matrix. Same answer, same
  query, and not one zero stored to get it.""")


# ── the exercise ─────────────────────────────────────────────────────────
head("4. The exercise, worked")

print("    Brutus AND Antony")
print()
print(f"    Brutus:  {' -> '.join(str(d) for d in index['Brutus'])}")
print(f"    Antony:  {' -> '.join(str(d) for d in index['Antony'])}")
print()
result = intersect(index["Brutus"], index["Antony"], trace=True)
print()
print(f"    result:  {' -> '.join(str(d) for d in result)}      ({names(result)})")
print("""
  The 4 and the 6 never match, and the walk passes them without ever
  comparing 4 against 1 or 2.""")


# ── OR ───────────────────────────────────────────────────────────────────
head("5. OR is the same walk, keeping everything")

def union(p1: list[int], p2: list[int]) -> list[int]:
    """Every document in either list, still sorted, each one output once."""
    result: list[int] = []
    i = j = 0
    while i < len(p1) and j < len(p2):
        if p1[i] == p2[j]:
            result.append(p1[i])
            i += 1
            j += 1
        elif p1[i] < p2[j]:
            result.append(p1[i])
            i += 1
        else:
            result.append(p2[j])
            j += 1
    # One list ran out; whatever is left in the other is all larger than
    # everything output so far, so it can be appended as it stands.
    return result + p1[i:] + p2[j:]


print(f"    Calpurnia OR Cleopatra    "
      f"{' '.join(map(str, union(index['Calpurnia'], index['Cleopatra'])))}"
      f"      ({names(union(index['Calpurnia'], index['Cleopatra']))})")
print("""
  Advance the smaller pointer and output it, rather than discarding it.
  Shared documents are output once.""")


# ── query optimisation ───────────────────────────────────────────────────
head("6. Start with the rarest term")

print("    Brutus AND Caesar AND Calpurnia")
print()
for order in [["Brutus", "Caesar", "Calpurnia"], ["Calpurnia", "Brutus", "Caesar"]]:
    # Intersect left to right in the given order and count how long each
    # intermediate result is. The answer is the same either way; the work is not.
    running = index[order[0]]
    sizes = [len(running)]
    for term in order[1:]:
        running = intersect(running, index[term])
        sizes.append(len(running))
    print(f"    {' -> '.join(order):42} list sizes {sizes}")

print()
print(f"    lengths:  " + "  ".join(f"{t} {len(index[t])}"
                                    for t in ["Brutus", "Caesar", "Calpurnia"]))
print("""
  Calpurnia is in one document. Intersecting it first leaves at most one
  candidate for everything after it, so the remaining work is trivial.
  Processing terms from rarest to commonest is a standard optimisation,
  and the postings list lengths are already stored, so the ordering is
  free to work out.""")

print()
