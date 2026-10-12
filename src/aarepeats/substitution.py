"""Substitution matrices as dense lookup tables indexed by byte code."""

from __future__ import annotations

from functools import lru_cache

import numpy as np


@lru_cache(maxsize=None)
def score_table(name: str = "BLOSUM62") -> np.ndarray:
    """Return a 256x256 ``int32`` table with ``table[ord(a), ord(b)]`` = score(a, b).

    Pairs absent from the matrix get the matrix's minimum score, so unknown
    residues are penalised rather than silently rewarded.
    """
    from Bio.Align import substitution_matrices  # imported lazily: optional at import time

    matrix = substitution_matrices.load(name)
    alphabet = matrix.alphabet
    values = np.asarray(matrix, dtype=np.int32)
    table = np.full((256, 256), values.min(), dtype=np.int32)
    codes = np.frombuffer(alphabet.encode("ascii"), dtype=np.uint8)
    table[np.ix_(codes, codes)] = values
    table.flags.writeable = False
    return table


def encode(seq: str) -> np.ndarray:
    """Encode a sequence as a ``uint8`` array of ASCII codes."""
    return np.frombuffer(seq.encode("ascii"), dtype=np.uint8)
