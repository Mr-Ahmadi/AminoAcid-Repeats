r"""Periodic repeat detection by Fourier analysis.

A sequence is mapped to one or more numeric signals: either a single
physico-chemical scale (e.g. hydropathy) or, by default, 20 binary *indicator*
signals, one per amino acid. The indicator encoding sees every residue
identity, whereas a scale signal carries almost no spectral power for repeats
whose residues have similar scale values: M/A and S/T each differ by only 0.1
on the Kyte-Doolittle scale, so (MA)n and (ST)n barely register globally.

**Local periodicity score.** For a window of length :math:`W_p` (a multiple of
:math:`p`) the fraction of the window's signal energy carried by the
harmonics :math:`k/p,\ k = 1..p-1` is

.. math::  S_p = \frac{\sum_{c}\sum_{k=1}^{p-1} |X_c(k/p)|^2}{W_p \sum_c \sum_j (x_{c,j} - \bar x_c)^2}

By Parseval's theorem :math:`0 \le S_p \le 1`, :math:`S_p = 1` for a perfectly
:math:`p`-periodic window, and its expectation for a random window is about
:math:`(p-1)/W_p`. We report the baseline-corrected excess score
:math:`E_p = (S_p - b_p)/(1 - b_p)`.

**Detection.** Windows where the smallest period with :math:`E_p \ge` threshold
propose candidate periods; candidates are verified and delimited by exact or
gapped self-matching (:mod:`aarepeats.tandem`). Period 1 (homopolymer runs) is
invisible to the DFT (frequency 1 aliases to 0) and is left to the entropy and
tandem modules.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .sequences import AMINO_ACIDS
from .tandem import TandemRepeat, find_tandem_repeats


def sequence_to_signal(seq: str, scale: dict[str, float] | None = None) -> np.ndarray:
    """Encode ``seq`` as an ``(n, channels)`` float array.

    With ``scale=None`` returns 20 indicator channels (one per standard amino
    acid; non-standard residues are all-zero rows). Otherwise returns a single
    z-scored channel of scale values (missing residues map to the scale mean).
    """
    if scale is None:
        lookup = {aa: i for i, aa in enumerate(AMINO_ACIDS)}
        out = np.zeros((len(seq), len(AMINO_ACIDS)))
        for j, aa in enumerate(seq):
            if aa in lookup:
                out[j, lookup[aa]] = 1.0
        return out
    mean = float(np.mean(list(scale.values())))
    x = np.array([scale.get(aa, mean) for aa in seq], dtype=float)
    sd = x.std()
    return ((x - x.mean()) / (sd if sd > 0 else 1.0))[:, None]


def power_spectrum(
    seq: str, scale: dict[str, float] | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """Whole-sequence power spectrum, summed over channels.

    Returns ``(frequency, power)`` for frequencies ``k / n``, ``k = 1 .. n // 2``
    (the DC term is dropped), with power normalised so that its mean over
    frequencies is 1 for a random sequence.
    """
    x = sequence_to_signal(seq, scale)
    x = x - x.mean(axis=0)
    n = x.shape[0]
    spectrum = np.abs(np.fft.rfft(x, axis=0)) ** 2
    power = spectrum.sum(axis=1)[1:]
    freq = np.fft.rfftfreq(n)[1:]
    total = (x**2).sum()
    return freq, power / (total if total > 0 else 1.0)


@dataclass(frozen=True)
class PeriodicityMap:
    """Local periodicity scores, ``score[p_index, i]`` for the window centred at ``i``."""

    periods: np.ndarray
    score: np.ndarray  # baseline-corrected excess score E_p, clipped to [0, 1]

    def best_period(self, threshold: float) -> np.ndarray:
        """Smallest period with score >= threshold at each position (0 = none)."""
        hit = self.score >= threshold
        first = np.argmax(hit, axis=0)
        return np.where(hit.any(axis=0), self.periods[first], 0)


def local_periodicity(
    seq: str,
    min_period: int = 2,
    max_period: int = 20,
    window: int = 12,
    scale: dict[str, float] | None = None,
) -> PeriodicityMap:
    """Compute the excess periodicity score E_p for every position and period.

    For period ``p`` the window length is ``W_p = p * max(2, window // p)`` so
    that every harmonic ``k/p`` falls exactly on a DFT bin.
    """
    if min_period < 2:
        raise ValueError("min_period must be >= 2 (period 1 aliases to frequency 0)")
    x = sequence_to_signal(seq, scale)
    n = x.shape[0]
    t = np.arange(n)
    zero = np.zeros((1, x.shape[1]))
    csum = np.vstack([zero, np.cumsum(x, axis=0)])
    csq = np.vstack([zero, np.cumsum(x**2, axis=0)])

    periods = np.arange(min_period, max_period + 1)
    score = np.zeros((periods.size, n))
    for row, p in enumerate(periods):
        w = p * max(2, window // p)
        if w > n:
            continue
        starts = np.arange(n - w + 1)
        sx = csum[starts + w] - csum[starts]
        energy = ((csq[starts + w] - csq[starts]) - sx**2 / w).sum(axis=1)  # centred
        harmonic = np.zeros(starts.size)
        for k in range(1, p):
            phase = np.exp(-2j * np.pi * k * t / p)[:, None]
            cz = np.vstack([zero.astype(complex), np.cumsum(x * phase, axis=0)])
            harmonic += (np.abs(cz[starts + w] - cz[starts]) ** 2).sum(axis=1)
        with np.errstate(divide="ignore", invalid="ignore"):
            s = np.where(energy > 1e-12, harmonic / (w * energy), 0.0)
        base = (p - 1) / w
        excess = np.clip((s - base) / (1 - base), 0.0, 1.0)
        centres = starts + w // 2
        score[row, centres] = excess
        # Extend edge values so every residue has a score.
        score[row, : centres[0]] = excess[0]
        score[row, centres[-1] + 1 :] = excess[-1]
    return PeriodicityMap(periods=periods, score=score)


def detect_periodic_repeats(
    seq: str,
    min_period: int = 2,
    max_period: int = 20,
    window: int = 12,
    threshold: float = 0.5,
    min_copies: float = 2.0,
    min_length: int = 6,
    max_gap: int = 0,
    scale: dict[str, float] | None = None,
) -> list[TandemRepeat]:
    """Detect tandem repeats whose period is supported by the local spectrum.

    1. Compute the local periodicity map (:func:`local_periodicity`).
    2. At each position, take the smallest period scoring >= ``threshold``
       (plus its divisors, which a periodic region necessarily also has).
    3. Verify those periods by self-matching (:func:`find_tandem_repeats`).
    4. Keep repeats whose span overlaps spectrally supported positions.
    """
    pmap = local_periodicity(seq, min_period, max_period, window, scale)
    best = pmap.best_period(threshold)
    # A p-periodic region is also periodic in multiples of its primitive period,
    # so offer every divisor too; tandem resolution keeps the shortest one.
    proposed = set(best[best > 0].tolist())
    candidates = sorted(
        {d for p in proposed for d in range(min_period, p + 1) if p % d == 0}
    )
    if not candidates:
        return []
    found = find_tandem_repeats(
        seq, periods=candidates, min_copies=min_copies, min_length=min_length, max_gap=max_gap
    )
    # Keep repeats whose span scores >= threshold at their period or a multiple of it.
    supported = []
    for rep in found:
        rows = (pmap.periods % rep.period) == 0
        if (pmap.score[rows, rep.start : rep.end] >= threshold).any():
            supported.append(rep)
    return supported


@dataclass(frozen=True)
class PeriodicRegion:
    """A stretch ``[start, end)`` whose composition is periodic with ``period``."""

    start: int
    end: int
    period: int
    mean_score: float

    @property
    def length(self) -> int:
        return self.end - self.start


def periodic_regions(
    seq: str,
    min_period: int = 2,
    max_period: int = 20,
    window: int = 36,
    threshold: float = 0.2,
    min_length: int = 30,
    scale: dict[str, float] | None = None,
) -> list[PeriodicRegion]:
    """Regions that are periodic *in pattern* without identical repeat units.

    Collagen's Gly-X-Y helix is the canonical example: only every third residue
    is conserved, so string matching sees no repeat, yet the spectrum has a
    sharp peak at period 3. Positions are labelled with the smallest period
    whose excess score is >= ``threshold``; runs of one label shorter than
    ``window // 2`` are absorbed into their neighbours before regions shorter
    than ``min_length`` are dropped.
    """
    pmap = local_periodicity(seq, min_period, max_period, window, scale)
    best = pmap.best_period(threshold)
    # Absorb short interruptions so that noise does not fragment a domain.
    edges = np.flatnonzero(np.diff(best)) + 1
    bounds = np.concatenate(([0], edges, [best.size]))
    for a, b in zip(bounds[:-1], bounds[1:]):
        if b - a < window // 2 and a > 0 and b < best.size and best[a - 1] == best[b]:
            best[a:b] = best[a - 1]
    regions = []
    edges = np.flatnonzero(np.diff(best)) + 1
    bounds = np.concatenate(([0], edges, [best.size]))
    for a, b in zip(bounds[:-1].tolist(), bounds[1:].tolist()):
        p = int(best[a])
        if p and b - a >= min_length:
            score = float(pmap.score[p - min_period, a:b].mean())
            regions.append(PeriodicRegion(a, b, p, score))
    return regions
