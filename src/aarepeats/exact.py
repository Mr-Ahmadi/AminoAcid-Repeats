"""Repeated words (exact or BLOSUM-approximate), possibly dispersed, with greedy tiling.

Candidate repeats are words that occur at least ``min_occurrences`` times. They
are grown one residue at a time: a word of length ``L`` can only repeat if its
length ``L-1`` prefix repeats, so only the start positions that survived length
``L-1`` are examined at length ``L`` and the search stops at the longest
repeated word. (The original implementation used a polynomial rolling hash in
``int64``, which overflows for words longer than ~8 residues and silently
merged unrelated words.)

In approximate mode each exact word is used as a seed and the whole sequence is
scanned for windows whose BLOSUM score against the seed is at least
``similarity`` times the seed's self-score.

Finally candidates are ranked by coverage (occurrences x length) and accepted
greedily so that no residue is assigned to two repeats.
"""

from __future__ import annotations

import time
from collections import defaultdict
from dataclasses import dataclass, field

import numpy as np

from .substitution import encode, score_table


@dataclass
class Repeat:
    """A word and its non-overlapping occurrence start positions."""

    word: str
    occurrences: list[int]
    matches: list[str] = field(default_factory=list)  # matched text (differs if approximate)

    @property
    def length(self) -> int:
        return len(self.word)

    @property
    def coverage(self) -> int:
        return self.length * len(self.occurrences)


def _non_overlapping(positions: list[int], length: int) -> list[int]:
    out, last_end = [], -1
    for pos in sorted(positions):
        if pos >= last_end:
            out.append(pos)
            last_end = pos + length
    return out


def repeated_words(seq: str, min_length: int = 2, min_occurrences: int = 2) -> dict[str, list[int]]:
    """All words of length >= ``min_length`` occurring >= ``min_occurrences`` times.

    Occurrence lists are non-overlapping (greedy left-to-right).
    """
    n = len(seq)
    active = list(range(n))
    words: dict[str, list[int]] = {}
    length = 1
    while active and length <= n // min_occurrences:
        groups: dict[str, list[int]] = defaultdict(list)
        for i in active:
            if i + length <= n:
                groups[seq[i : i + length]].append(i)
        active = []
        for word, pos in groups.items():
            if len(pos) < min_occurrences:  # overlapping count is an upper bound
                continue
            active.extend(pos)
            if length >= min_length:
                occ = _non_overlapping(pos, length)
                if len(occ) >= min_occurrences:
                    words[word] = occ
        length += 1
    return words


def _approximate_occurrences(
    codes: np.ndarray, seed: str, table: np.ndarray, similarity: float
) -> list[int]:
    L = len(seed)
    if len(codes) < L:
        return []
    s = encode(seed)
    windows = np.lib.stride_tricks.sliding_window_view(codes, L)
    scores = table[windows, s].sum(axis=1)
    perfect = table[s, s].sum()
    if perfect <= 0:
        return []
    hits = np.flatnonzero(scores >= similarity * perfect)
    return _non_overlapping(hits.tolist(), L)


def find_all_repeats(
    seq: str,
    min_length: int = 2,
    min_occurrences: int = 2,
    approximate: bool = False,
    similarity: float = 0.85,
    matrix: str = "BLOSUM62",
) -> tuple[list[Repeat], float]:
    """Detect repeated words and tile them greedily without overlap.

    Parameters
    ----------
    approximate:
        If True, also accept windows whose BLOSUM score against the word is
        at least ``similarity`` times the word's self-score.

    Returns
    -------
    ``(repeats, seconds)``, repeats ordered by decreasing coverage.
    """
    t0 = time.perf_counter()
    candidates = []
    words = repeated_words(seq, min_length, min_occurrences)
    if approximate:
        table = score_table(matrix)
        codes = encode(seq)
        for word in words:
            occ = _approximate_occurrences(codes, word, table, similarity)
            if len(occ) >= min_occurrences:
                candidates.append(Repeat(word, occ))
    else:
        candidates = [Repeat(w, occ) for w, occ in words.items()]

    candidates.sort(key=lambda r: (r.coverage, r.length, r.word), reverse=True)
    used = np.zeros(len(seq), dtype=bool)
    accepted = []
    for cand in candidates:
        L = cand.length
        free = [p for p in cand.occurrences if not used[p : p + L].any()]
        if len(free) >= min_occurrences:
            for p in free:
                used[p : p + L] = True
            accepted.append(Repeat(cand.word, free, [seq[p : p + L] for p in free]))
    return accepted, time.perf_counter() - t0
