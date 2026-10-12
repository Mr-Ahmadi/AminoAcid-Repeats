"""Tandem (adjacent, periodic) repeat scanning for a given set of periods.

For period ``p`` the sequence is compared with itself shifted by ``p``:
``match[i] = seq[i] == seq[i + p]``. A run of ``k`` consecutive matches means
``seq[i : i + k + p]`` is ``p``-periodic, i.e. ``(k + p) / p`` tandem copies of
a ``p``-residue unit. Short mismatch gaps may be bridged to tolerate
imperfect repeats. Overlapping calls are resolved in favour of longer
regions and, for equal length, shorter (primitive) periods, so ``(MA)n`` is
reported with period 2 rather than 4 or 6.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Iterable

import numpy as np

from .substitution import encode


@dataclass(frozen=True)
class TandemRepeat:
    """A tandem repeat occupying ``[start, end)`` with repeat unit length ``period``."""

    start: int
    end: int
    period: int
    unit: str
    identity: float  # fraction of residues matching the consensus unit at their phase
    sequence: str

    @property
    def length(self) -> int:
        return self.end - self.start

    @property
    def copies(self) -> float:
        return self.length / self.period


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    padded = np.concatenate(([False], mask, [False]))
    edges = np.flatnonzero(np.diff(padded.astype(np.int8)))
    return list(zip(edges[::2].tolist(), edges[1::2].tolist()))


def _consensus_unit(seq: str, start: int, end: int, period: int) -> str:
    """Majority residue at each phase of the repeat, starting at ``start``."""
    return "".join(
        Counter(seq[start + k : end : period]).most_common(1)[0][0] for k in range(period)
    )


def _identity(seq: str, start: int, end: int, unit: str) -> float:
    """Fraction of residues in ``seq[start:end]`` equal to the consensus unit at their phase."""
    p = len(unit)
    return sum(seq[i] == unit[(i - start) % p] for i in range(start, end)) / (end - start)


def _primitive(unit: str) -> str:
    """Shortest ``u`` with ``unit == u * k`` (``"PPPP"`` -> ``"P"``, ``"MAMA"`` -> ``"MA"``)."""
    p = len(unit)
    for q in range(1, p):
        if p % q == 0 and unit == unit[:q] * (p // q):
            return unit[:q]
    return unit


def _make(seq: str, start: int, end: int, period: int) -> TandemRepeat:
    unit = _primitive(_consensus_unit(seq, start, end, period))
    period = len(unit)
    return TandemRepeat(start, end, period, unit, _identity(seq, start, end, unit), seq[start:end])


def _continues(
    seq: str, pa: int, pb: int, a: int, b: int, period: int, min_identity: float
) -> bool:
    """Whether match run ``[a, b)`` extends the repeat built from run ``[pa, pb)``.

    The new run must itself match the current consensus unit (in phase), and
    the merged region must keep ``min_identity`` overall.
    """
    unit = _consensus_unit(seq, pa, pb + period, period)
    new = range(a, b + period)
    if sum(seq[i] == unit[(i - pa) % period] for i in new) / len(new) < min_identity:
        return False
    return _make(seq, pa, b + period, period).identity >= min_identity


def _scan_period(
    seq: str, codes: np.ndarray, period: int, max_gap: int, min_identity: float, min_span: int
) -> list[TandemRepeat]:
    """Tandem repeats of one period. A match run ``[a, b)`` spans ``[a, b + period)``."""
    match = codes[:-period] == codes[period:]
    runs = _runs(match)
    if max_gap > 0 and runs:
        # Bridge short mismatch gaps, but only if the merged region stays
        # consistent with a single consensus unit (so Q21 followed by P11 is
        # not merged into one "Q" repeat).
        # One substitution breaks the match array at i - period and at i; for
        # period 1 those two mismatches are adjacent, so allow twice the gap.
        allowed = 2 * max_gap if period == 1 else max_gap
        merged = [runs[0]]
        for a, b in runs[1:]:
            pa, pb = merged[-1]
            if a - pb <= allowed and _continues(seq, pa, pb, a, b, period, min_identity):
                merged[-1] = (pa, b)
            else:
                merged.append((a, b))
        runs = merged
    out = []
    for a, b in runs:
        if b + period - a < min_span:
            continue
        rep = _make(seq, a, b + period, period)
        if rep.identity >= min_identity:
            out.append(rep)
    return out


def _qualifies(rep: TandemRepeat, min_copies: float, min_length: int) -> bool:
    return rep.copies >= min_copies and rep.length >= min_length


def find_tandem_repeats(
    seq: str,
    periods: Iterable[int] | None = None,
    min_copies: float = 2.0,
    min_length: int = 6,
    max_gap: int = 0,
    min_identity: float = 0.8,
    max_period: int = 50,
) -> list[TandemRepeat]:
    """Scan ``seq`` for tandem repeats with the given unit lengths.

    Parameters
    ----------
    periods:
        Unit lengths to test (default: 1 .. ``max_period``).
    min_copies:
        Minimum number of (possibly fractional) copies of the unit.
    min_length:
        Minimum total length of the repeat region in residues.
    max_gap:
        Bridge mismatch gaps of up to this many positions in the shifted
        comparison (0 = exact repeats; 1 tolerates isolated substitutions).
    min_identity:
        Minimum fraction of residues matching the consensus unit (only binds
        when ``max_gap > 0``; exact runs have identity 1).
    max_period:
        Largest unit length tested when ``periods`` is not given.

    Returns
    -------
    Non-overlapping repeats sorted by start position.
    """
    codes = encode(seq)
    if periods is None:
        periods = range(1, min(max_period, len(seq) // 2) + 1)
    candidates: list[TandemRepeat] = []
    for p in sorted(set(int(p) for p in periods)):
        if 1 <= p < len(seq):
            candidates += [
                r
                for r in _scan_period(seq, codes, p, max_gap, min_identity, min_length)
                if _qualifies(r, min_copies, min_length)
            ]

    # Longest first; for ties prefer the primitive (shortest) period.
    candidates.sort(key=lambda r: (-r.length, r.period, r.start))
    taken = np.zeros(len(seq), dtype=bool)
    accepted: list[TandemRepeat] = []
    for rep in candidates:
        free = ~taken[rep.start : rep.end]
        if free.all():
            accepted.append(rep)
            taken[rep.start : rep.end] = True
            continue
        # Keep the largest non-overlapping stretch if it still qualifies.
        best = max(_runs(free), key=lambda ab: ab[1] - ab[0], default=None)
        if best is None:
            continue
        a, b = rep.start + best[0], rep.start + best[1]
        if b - a < max(min_length, rep.period):
            continue
        sub = _make(seq, a, b, rep.period)
        if _qualifies(sub, min_copies, min_length) and sub.identity >= min_identity:
            accepted.append(sub)
            taken[a:b] = True
    return sorted(accepted, key=lambda r: r.start)
