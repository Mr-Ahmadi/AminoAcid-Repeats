"""Low-complexity regions (LCRs) from sliding-window Shannon entropy.

The detector follows the two-threshold scheme of SEG (Wootton & Federhen, 1993):
windows whose entropy is at or below a *trigger* threshold seed a region, which
then grows through neighbouring windows whose entropy stays at or below a more
permissive *extension* threshold. The defaults (window 12, trigger 2.2 bits,
extension 2.5 bits) are SEG's standard parameters.
"""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Region:
    """A half-open, 0-based interval ``[start, end)`` of a sequence."""

    start: int
    end: int
    sequence: str
    entropy: float

    @property
    def length(self) -> int:
        return self.end - self.start


@dataclass(frozen=True)
class LCRResult:
    """Output of :func:`detect_lcr`."""

    window: int
    entropies: np.ndarray  # entropy of window starting at i, length n - window + 1
    regions: list[Region]

    @property
    def centers(self) -> np.ndarray:
        """Sequence coordinate of each window's centre (for plotting)."""
        return np.arange(self.entropies.size) + (self.window - 1) / 2


def shannon_entropy(seq: str) -> float:
    """Shannon entropy of the residue composition of ``seq``, in bits."""
    n = len(seq)
    if n == 0:
        return 0.0
    return -sum((c / n) * math.log2(c / n) for c in Counter(seq).values())


def sliding_entropy(seq: str, window: int) -> np.ndarray:
    """Entropy (bits) of every length-``window`` substring, by start position.

    Vectorised with cumulative one-hot counts, so it costs O(n * alphabet).
    """
    n = len(seq)
    if not 1 <= window <= n:
        raise ValueError(f"window must be in [1, {n}], got {window}")
    letters = sorted(set(seq))
    index = np.array([letters.index(c) for c in seq])
    onehot = np.zeros((n + 1, len(letters)), dtype=np.int32)
    onehot[np.arange(1, n + 1), index] = 1
    cumulative = onehot.cumsum(axis=0)
    counts = cumulative[window:] - cumulative[:-window]  # (n - window + 1, k)
    p = counts / window
    with np.errstate(divide="ignore", invalid="ignore"):
        terms = np.where(p > 0, p * np.log2(p), 0.0)
    return -terms.sum(axis=1) + 0.0  # + 0.0 turns -0.0 into 0.0


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """Half-open ``[start, end)`` intervals where a boolean array is True."""
    padded = np.concatenate(([False], mask, [False]))
    edges = np.flatnonzero(np.diff(padded.astype(np.int8)))
    return list(zip(edges[::2].tolist(), edges[1::2].tolist()))


def _trim_singletons(seq: str, start: int, end: int) -> tuple[int, int]:
    """Drop region ends whose residue occurs only once in the region.

    Window unions overshoot true boundaries by up to ``window - 1`` residues;
    a flanking residue that appears exactly once does not belong to the
    low-complexity composition and is removed.
    """
    while end - start > 1:
        counts = Counter(seq[start:end])
        if counts[seq[start]] == 1:
            start += 1
        elif counts[seq[end - 1]] == 1:
            end -= 1
        else:
            break
    return start, end


def detect_lcr(
    seq: str,
    window: int = 12,
    trigger: float = 2.2,
    extension: float = 2.5,
    min_length: int | None = None,
) -> LCRResult:
    """Find low-complexity regions with SEG-style trigger/extension thresholds.

    Parameters
    ----------
    seq:
        Protein sequence.
    window:
        Sliding-window length (clipped to the sequence length).
    trigger:
        A window with entropy <= ``trigger`` bits seeds a region.
    extension:
        Neighbouring windows with entropy <= ``extension`` bits extend it.
    min_length:
        Discard regions shorter than this (default: ``window // 2``).
    """
    if extension < trigger:
        raise ValueError("extension threshold must be >= trigger threshold")
    window = max(1, min(window, len(seq)))
    if min_length is None:
        min_length = max(1, window // 2)

    entropies = sliding_entropy(seq, window)
    seeded = entropies <= trigger
    covered = np.zeros(len(seq), dtype=bool)
    for a, b in _runs(entropies <= extension):
        if seeded[a:b].any():
            covered[a : b - 1 + window] = True

    regions = []
    for a, b in _runs(covered):
        a, b = _trim_singletons(seq, a, b)
        if b - a >= min_length:
            regions.append(Region(a, b, seq[a:b], shannon_entropy(seq[a:b])))
    return LCRResult(window=window, entropies=entropies, regions=regions)
