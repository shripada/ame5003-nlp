"""A mini search engine over the Reuters corpus — project 1, AME 5053.

The pieces, in the order you build them:

    extract    pull headline, body, amounts and dates out of raw text   (sessions 1-2)
    pipeline   raw text -> index terms                                  (sessions 3-5)
    matrix     the term-document incidence matrix                       (session 8)
    index      the inverted index and its postings merges               (session 9)
    ranking    TF-IDF weights and ranked retrieval                      (session 10)
    evaluate   precision@10 and the ablation                            (stage 5)

`corpus`, `evaluate`'s harness and `cli` are given to you. Everything else has
stubs to fill in. Run `pytest` to find out how you are doing.
"""

