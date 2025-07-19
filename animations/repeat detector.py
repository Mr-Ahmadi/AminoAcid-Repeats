import time
import os
from typing import List, Tuple, Dict, Optional
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter
from matplotlib.patches import Rectangle
from matplotlib.colors import Normalize
from Bio.Align import substitution_matrices

# Optional: use Numba for JIT-compiling inner loops.
try:
    from numba import njit
except ImportError:
    def njit(func=None, **kwargs):
        if func is None:
            return lambda func: func
        return func

plt.style.use('seaborn-darkgrid' if 'seaborn-darkgrid' in plt.style.available else 'ggplot')


def preprocess_substitution_matrix(substitution_matrix_name: str, save_dir: str = '.') -> np.ndarray:
    """
    Load and preprocess a substitution matrix into a 256x256 score array.
    """
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"{substitution_matrix_name}_score_matrix.npy")
    if os.path.exists(save_path):
        return np.load(save_path)
    matrix = substitution_matrices.load(substitution_matrix_name)
    score_matrix = np.full((256, 256), -9999, dtype=np.int16)
    for (a, b), score in matrix.items():
        score_matrix[ord(a), ord(b)] = score
        score_matrix[ord(b), ord(a)] = score  # assume symmetry
    np.save(save_path, score_matrix)
    return score_matrix


def rolling_hash_vectorized(ord_s: np.ndarray, L: int, base: int, mod: int) -> np.ndarray:
    """
    Compute rolling hashes for an array of character codes using vectorized operations.
    """
    n = ord_s.shape[0]
    if n < L:
        return np.array([], dtype=np.int64)
    shape = (n - L + 1, L)
    strides = (ord_s.strides[0], ord_s.strides[0])
    windows = np.lib.stride_tricks.as_strided(ord_s, shape=shape, strides=strides)
    powers = np.power(base, np.arange(L-1, -1, -1), dtype=np.int64) % mod
    h = (windows.astype(np.int64) * powers).sum(axis=1) % mod
    return h


@njit
def imperfect_check_loop(ord_s: np.ndarray, valid_indices: np.ndarray, L: int, 
                           score_matrix: np.ndarray, relative_threshold: float, tolerance: float) -> Tuple[int, int]:
    """
    (Not currently integrated into the main loop)
    Check for approximate matches using the substitution matrix.
    Returns a tuple of indices (i, j) for a candidate pair, or (-1, -1) if not found.
    """
    num = valid_indices.shape[0]
    perfect_scores = np.empty(num, dtype=np.int64)
    for k in range(num):
        i = valid_indices[k]
        ps = 0
        for pos in range(L):
            ps += score_matrix[ord_s[i+pos], ord_s[i+pos]]
        perfect_scores[k] = ps

    for a in range(num):
        i = valid_indices[a]
        b_start = np.searchsorted(valid_indices, i + L)
        for b in range(b_start, num):
            j = valid_indices[b]
            score_ij = 0
            for pos in range(L):
                score_ij += score_matrix[ord_s[i+pos], ord_s[j+pos]]
            if perfect_scores[a] > 0 and score_ij >= (relative_threshold - tolerance) * perfect_scores[a]:
                return i, j
    return -1, -1


def find_occurrences_masked(s: str, sub: str, mask: List[bool]) -> List[int]:
    """
    Exact matching: finds all non-overlapping occurrences of 'sub' in 's'
    where the substring is not masked.
    """
    L = len(sub)
    n = len(s)
    occurrences = []
    i = 0
    while i <= n - L:
        if s[i:i+L] == sub and not any(mask[i:i+L]):
            occurrences.append(i)
            i += L  # enforce non-overlap
        else:
            i += 1
    return occurrences


def find_occurrences_approx(s: str, sub: str, mask: List[bool], score_matrix: np.ndarray, 
                            tolerance: float, relative_threshold: float) -> List[int]:
    """
    Approximate matching: finds occurrences of 'sub' in 's' using the substitution matrix.
    """
    L = len(sub)
    n = len(s)
    if L == 0 or n < L:
        return []
    mask_arr = np.array(mask, dtype=bool)
    mask_cumsum = np.concatenate(([0], np.cumsum(mask_arr)))
    valid_windows = np.where((mask_cumsum[L:] - mask_cumsum[:-L]) == 0)[0]
    if valid_windows.size == 0:
        return []
    sub_ord = np.array([ord(c) for c in sub], dtype=np.uint8)
    ord_s = np.array([ord(c) for c in s], dtype=np.uint8)
    windows = ord_s[valid_windows[:, None] + np.arange(L)]
    scores = np.sum(score_matrix[windows, sub_ord], axis=1)
    perfect_score = np.sum(score_matrix[sub_ord, sub_ord])
    ratios = scores / perfect_score
    candidates = valid_windows[ratios >= (relative_threshold - tolerance)]
    occurrences = []
    last = -L
    for pos in candidates:
        if pos >= last + L:
            occurrences.append(pos)
            last = pos
    return occurrences


def find_all_repeats(s: str, min_length: int = 2, min_occurrences: int = 2,
                     substitution_matrix_name: Optional[str] = None,
                     relative_threshold: float = 1.0, tolerance: Optional[float] = None) -> Tuple[List[Dict], float]:
    """
    Detect candidate repeats in string 's'. For each candidate substring of length L (from min_length to n//2)
    that appears at least min_occurrences times (non-overlapping), record its coverage.
    A greedy selection then chooses repeats that do not overlap.
    
    If a substitution_matrix_name is provided (with non-None tolerance), approximate matching is used.
    Otherwise, exact matching is performed.
    """
    n = len(s)
    start_time = time.perf_counter()
    candidates = []
    base = max(256, len(set(s)))
    mod = (1 << 61) - 1
    ord_s = np.array([ord(c) for c in s], dtype=np.uint8)
    
    # If using approximate matching, load the substitution matrix.
    score_matrix = None
    if substitution_matrix_name is not None:
        if tolerance is None:
            tolerance = 0.0  # default tolerance if not provided
        score_matrix = preprocess_substitution_matrix(substitution_matrix_name)
    
    for L in range(min_length, n // 2 + 1):
        valid_indices = np.arange(0, n - L + 1)
        hashes = rolling_hash_vectorized(ord_s, L, base, mod)
        unique_hashes = np.unique(hashes)
        for h in unique_hashes:
            group = valid_indices[hashes == h]
            if group.size < min_occurrences:
                continue
            candidate_repeat = s[group[0]:group[0]+L]
            # Use approximate matching if a substitution matrix is provided.
            if substitution_matrix_name is None:
                occs = find_occurrences_masked(s, candidate_repeat, [False] * n)
            else:
                occs = find_occurrences_approx(s, candidate_repeat, [False] * n,
                                               score_matrix, tolerance, relative_threshold)
            if len(occs) < min_occurrences:
                continue
            coverage = len(occs) * L
            candidates.append({
                'repeat': candidate_repeat,
                'occurrences': occs,
                'coverage': coverage,
                'length': L
            })
    
    # Sort candidates by coverage and length (both descending)
    candidates.sort(key=lambda x: (x['coverage'], x['length']), reverse=True)
    
    # Greedy selection: choose repeats whose occurrences do not overlap.
    used = [False] * n
    final_repeats = []
    for cand in candidates:
        L = cand['length']
        valid_occs = []
        for pos in cand['occurrences']:
            if not any(used[pos:pos+L]):
                valid_occs.append(pos)
        if len(valid_occs) >= min_occurrences:
            for pos in valid_occs:
                for i in range(pos, pos+L):
                    used[i] = True
            cand['occurrences'] = valid_occs
            final_repeats.append(cand)
    
    total_time = time.perf_counter() - start_time
    return final_repeats, total_time


def animate_repeats(s: str, repeats: List[Dict], save_filename: str = "output.mp4", dpi: int = 300) -> None:
    """
    Create an animation visualizing the detected repeats.
    """
    cmap = plt.get_cmap('coolwarm')
    vmax = max((r['length'] for r in repeats), default=20)
    norm = Normalize(vmin=2, vmax=vmax)
    x_padding = 0.05 * len(s)
    y_padding = 0.1

    fig, ax = plt.subplots(figsize=(16, 6), facecolor='#f5f5f5')
    ax.set_facecolor('#f5f5f5')
    ax.set(xlim=(-x_padding, len(s)+x_padding), ylim=(-y_padding, 2+y_padding),
           yticks=[], title="Repeat Detection Progress", xlabel="Position")
    writer = FFMpegWriter(fps=5, bitrate=5000)
    writer.setup(fig, save_filename, dpi=dpi)
    
    # Draw initial frame.
    ax.clear()
    ax.set_facecolor('#f5f5f5')
    ax.hlines(1, -x_padding, len(s)+x_padding, colors='#34495e', lw=4, alpha=0.8)
    ax.text(0.01, 0.85, "Starting Repeat Detection Animation", fontsize=12,
            transform=ax.transAxes, bbox=dict(facecolor='white', alpha=0.8))
    writer.grab_frame()
    
    cumulative_repeats = [(rep['repeat'], rep['occurrences']) for rep in repeats]
    
    for idx, (repeat, occs) in enumerate(cumulative_repeats):
        ax.clear()
        ax.set_facecolor('#f5f5f5')
        ax.hlines(1, -x_padding, len(s)+x_padding, colors='#34495e', lw=4, alpha=0.8)
        for rep_inner, occs_inner in cumulative_repeats[:idx+1]:
            L = len(rep_inner)
            clr = cmap(norm(L))
            for pos in occs_inner:
                rect = Rectangle((pos, 0.6), L, 0.8, color=clr, alpha=0.9, ec='black', lw=2)
                ax.add_patch(rect)
        info_text = f"Detected repeat '{repeat}' at positions {occs}"
        ax.text(0.01, 0.85, info_text, fontsize=12, transform=ax.transAxes,
                bbox=dict(facecolor='white', alpha=0.8))
        writer.grab_frame()
    
    writer.finish()
    plt.close()
    print(f"Animation saved to {save_filename}")


def main():
    s = ("MASACDEFGHIMAMAMAMAMAMAMAMAMAAPRTEINSEQENCEAAQAAAQHMAMAHHHTESTSTSTSTST"
        "SSGSGSGSGSWWWWWGSGSACDEFGHIKACDEFGHIKACDEFGHIKACDEFGHIK"
        "MAMAMAMAMAMACDEFGHIKACDEFGHIKGSGSGSGSTESTTESTTESTACDEFGHIK"
        "APRTEINSEQENCEAAAAAAACDEFGHIK")
    min_length = 2
    min_occurrences = 3

    # --- Choose Matching Mode ---
    # To use exact matching, leave substitution_matrix_name as None:
    # repeats, total_time = find_all_repeats(s, min_length=min_length, min_occurrences=min_occurrences)
    
    # To use approximate matching, provide the substitution matrix and parameters:
    relative_threshold = 0.9  # e.g., require 90% of the perfect match score
    tolerance = 0.05          # allow a 5% tolerance
    substitution_matrix_name = "BLOSUM62"
    repeats, total_time = find_all_repeats(s, min_length=min_length, min_occurrences=min_occurrences,
                                            substitution_matrix_name=substitution_matrix_name,
                                            relative_threshold=relative_threshold, tolerance=tolerance)

    if repeats:
        print("\nRepeats found:")
        for idx, rep in enumerate(repeats):
            print(f"Candidate {idx+1}: '{rep['repeat']}' at positions {rep['occurrences']} (coverage={rep['coverage']})")
    else:
        print("No repeats found.")
    print(f"\nTotal time: {total_time:.2f}s")

    animate_repeats(s, repeats)


if __name__ == "__main__":
    main()
