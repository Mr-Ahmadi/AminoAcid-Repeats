"""Internal repeats by Smith-Waterman self-alignment with Waterman-Eggert declumping.

The sequence is aligned against itself using BLOSUM62 and a linear gap
penalty. Only the upper triangle ``j - i >= min_offset`` is filled, so the
trivial main diagonal and the mirror-image lower triangle are excluded.

After the best local alignment is reported, every cell on its path is forced to
zero and the matrix is recomputed (Waterman & Eggert, 1987). This removes the
"shadow" alignments that share cells with an already reported one, so each
reported alignment is a genuinely distinct repeat relationship.

Each row of the recurrence

    H[i, j] = max(0, H[i-1, j-1] + s(a_i, a_j), H[i-1, j] + g, H[i, j-1] + g)

is vectorised: the horizontal term is a running maximum,
``H[i, j] = g*j + cummax_k<=j (E[k] - g*k)``, segmented at masked cells.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from .substitution import encode, score_table

@dataclass(frozen=True)
class SelfAlignment:
    """A local alignment between two segments ``[start1, end1)`` and ``[start2, end2)``."""

    score: float
    start1: int
    end1: int
    start2: int
    end2: int
    aligned1: str
    aligned2: str
    zscore: float | None = None

    @property
    def offset(self) -> int:
        """Diagonal offset ``start2 - start1`` (the repeat period for tandem repeats)."""
        return self.start2 - self.start1

    @property
    def identity(self) -> float:
        pairs = list(zip(self.aligned1, self.aligned2))
        return sum(a == b for a, b in pairs) / len(pairs)

    @property
    def kind(self) -> str:
        """``"tandem"`` if the two copies touch or overlap, otherwise ``"dispersed"``."""
        return "tandem" if self.start2 <= self.end1 else "dispersed"


def _fill(
    sub: np.ndarray, gap: float, blocked: np.ndarray, H: np.ndarray, first_row: int
) -> None:
    """Fill ``H`` (shape ``(n+1, n+1)``) in place from ``first_row`` onwards."""
    n = sub.shape[0]
    j = np.arange(1, n + 1, dtype=float)
    # Offset that keeps the running maximum from crossing a masked cell; sized
    # to exceed the value range while keeping float rounding negligible.
    big = 4.0 * (abs(gap) + np.abs(sub).max() + 1.0) * (n + 1)
    for i in range(max(1, first_row), n + 1):
        prev = H[i - 1]
        e = np.maximum(prev[:-1] + sub[i - 1], prev[1:] + gap)
        e = np.maximum(e, 0.0)
        mask = blocked[i, 1:]
        e[mask] = 0.0
        seg = np.cumsum(mask)
        v = e - gap * j + seg * big
        h = np.maximum.accumulate(v) - seg * big + gap * j
        h = np.maximum(h, e)
        h[mask] = 0.0
        H[i, 1:] = h


def _traceback(
    H: np.ndarray, sub: np.ndarray, gap: float, i: int, j: int, seq: str
) -> tuple[list[tuple[int, int]], str, str]:
    path, a1, a2 = [], [], []
    while i > 0 and j > 0 and H[i, j] > 0:
        path.append((i, j))
        h = H[i, j]
        if np.isclose(h, H[i - 1, j - 1] + sub[i - 1, j - 1]):
            a1.append(seq[i - 1]); a2.append(seq[j - 1]); i -= 1; j -= 1
        elif np.isclose(h, H[i - 1, j] + gap):
            a1.append(seq[i - 1]); a2.append("-"); i -= 1
        else:
            a1.append("-"); a2.append(seq[j - 1]); j -= 1
    return path[::-1], "".join(reversed(a1)), "".join(reversed(a2))


def _overlap(a0: int, a1: int, b0: int, b1: int) -> float:
    """Overlap of two intervals as a fraction of the shorter one."""
    inter = min(a1, b1) - max(a0, b0)
    return max(0, inter) / max(1, min(a1 - a0, b1 - b0))


def _redundant(a: "SelfAlignment", b: "SelfAlignment", max_overlap: float) -> bool:
    same = _overlap(a.start1, a.end1, b.start1, b.end1) > max_overlap and \
        _overlap(a.start2, a.end2, b.start2, b.end2) > max_overlap
    swapped = _overlap(a.start1, a.end1, b.start2, b.end2) > max_overlap and \
        _overlap(a.start2, a.end2, b.start1, b.end1) > max_overlap
    return same or swapped


def _mask_lcr(seq: str) -> str:
    from .entropy import detect_lcr

    chars = list(seq)
    for region in detect_lcr(seq).regions:
        chars[region.start : region.end] = "X" * region.length
    return "".join(chars)


def _shuffle_unmasked(seq: str, rng: np.random.Generator) -> str:
    """Permute residues among unmasked positions; masked ``X`` positions stay put."""
    chars = np.array(list(seq))
    free = np.flatnonzero(chars != "X")
    chars[free] = rng.permutation(chars[free])
    return "".join(chars)


def _best_score(seq: str, matrix: str, gap: float, min_offset: int) -> float:
    n = len(seq)
    codes = encode(seq)
    sub = score_table(matrix)[np.ix_(codes, codes)].astype(float)
    ii, jj = np.indices((n + 1, n + 1))
    blocked = (jj - ii) < min_offset
    H = np.zeros((n + 1, n + 1))
    _fill(sub, gap, blocked, H, 1)
    return float(H.max())


def find_internal_repeats(
    seq: str,
    gap: float = -8.0,
    min_score: float = 25.0,
    max_alignments: int = 10,
    min_offset: int = 1,
    matrix: str = "BLOSUM62",
    n_shuffles: int = 0,
    seed: int = 0,
    max_overlap: float = 0.5,
    mask_low_complexity: bool = False,
) -> list[SelfAlignment]:
    """Report up to ``max_alignments`` non-redundant local self-alignments.

    Parameters
    ----------
    gap:
        Linear gap penalty (negative).
    min_score:
        Stop when the best remaining alignment scores below this.
    min_offset:
        Ignore cells with ``j - i < min_offset`` (1 excludes only the main diagonal).
    n_shuffles:
        If > 0, estimate a z-score for each alignment against the best
        self-alignment score of ``n_shuffles`` composition-preserving shuffles
        of the sequence (with masked positions held fixed). The null describes the *best*
        alignment, so z-scores of lower-ranked alignments are conservative.
    max_overlap:
        Skip an alignment if both of its segments overlap the corresponding
        segments of an already reported alignment by more than this fraction
        (register-shifted copies of the same repeat relationship).
    mask_low_complexity:
        Replace SEG low-complexity regions with ``X`` before scoring, as BLAST
        and RADAR do, so that compositionally biased stretches do not dominate.
    """
    if gap >= 0:
        raise ValueError("gap penalty must be negative")
    n = len(seq)
    if n < 2:
        return []
    scored = _mask_lcr(seq) if mask_low_complexity else seq
    codes = encode(scored)
    sub = score_table(matrix)[np.ix_(codes, codes)].astype(float)
    ii, jj = np.indices((n + 1, n + 1))
    blocked = (jj - ii) < min_offset
    blocked[0, :] = blocked[:, 0] = True
    H = np.zeros((n + 1, n + 1))
    _fill(sub, gap, blocked, H, 1)

    results: list[SelfAlignment] = []
    for _ in range(20 * max_alignments):
        if len(results) >= max_alignments:
            break
        i, j = np.unravel_index(int(np.argmax(H)), H.shape)
        score = float(H[i, j])
        if score < min_score:
            break
        path, a1, a2 = _traceback(H, sub, gap, i, j, seq)
        rows = [p[0] for p in path]
        cols = [p[1] for p in path]
        candidate = SelfAlignment(
            score=score,
            start1=int(min(rows)) - 1, end1=int(max(rows)),
            start2=int(min(cols)) - 1, end2=int(max(cols)),
            aligned1=a1, aligned2=a2,
        )
        if not any(_redundant(candidate, r, max_overlap) for r in results):
            results.append(candidate)
        for r, c in path:
            blocked[r, c] = True
        _fill(sub, gap, blocked, H, min(rows))

    if n_shuffles > 0 and results:
        rng = np.random.default_rng(seed)
        null = np.array([
            _best_score(_shuffle_unmasked(scored, rng), matrix, gap, min_offset)
            for _ in range(n_shuffles)
        ])
        mu, sd = null.mean(), null.std(ddof=1) if n_shuffles > 1 else 1.0
        results = [replace(r, zscore=(r.score - mu) / sd if sd > 0 else float("inf")) for r in results]
    return results


def windowed_dotplot(seq: str, window: int = 10, other: str | None = None) -> np.ndarray:
    """Number of identities in a length-``window`` diagonal stretch ending at ``(i, j)``.

    This is the classic filtered dot plot (Gibbs & McIntyre, 1970); compare
    against a threshold such as ``0.7 * window`` for display.
    """
    a = encode(seq)
    b = encode(other if other is not None else seq)
    match = (a[:, None] == b[None, :]).astype(np.int32)
    diag = np.zeros((len(a) + 1, len(b) + 1), dtype=np.int32)
    for i in range(1, len(a) + 1):
        diag[i, 1:] = match[i - 1] + diag[i - 1, :-1]
    out = np.zeros_like(match)
    w = window
    out[w - 1 :, w - 1 :] = diag[w:, w:] - diag[:-w, :-w]
    return out
