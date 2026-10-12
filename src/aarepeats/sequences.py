"""Sequence input, validation and the example sequences used throughout the project."""

from __future__ import annotations

from pathlib import Path
from typing import Iterator

#: The 20 standard amino acids in one-letter code.
AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"

#: IUPAC one-letter codes accepted in input (standard + ambiguity/rare residues).
VALID_CODES = frozenset(AMINO_ACIDS + "BJOUXZ")

#: Synthetic test sequences used in the original notebooks. Each one mixes
#: perfect tandem repeats (MA)n, (ST)n, (SG)n, homopolymer runs (A)n, (W)n and
#: dispersed repeats (ACDEFGHIK, APRTEINSEQENCE) with non-repetitive spacer.
EXAMPLES = {
    "toy": "MAMAMAMAMAMAPRTEINSEQENCEAAAAAAHHHHTESTSTSTSTSTSSGSGSGSGSWWWWWGSGS",
    "toy_dispersed": "MAMAMAMAMAMAPRTEINSEQENCEAAAAAAHHHHTESTSTSTSTSTSMAMAMAMASGSGSGSGSWWWWWMAMAMAMAGSGS",
    "toy_long": (
        "MASACDEFGHIMAMAMAMAMAMAMAMAMAAPRTEINSEQENCEAAQAAAQHMAMAHHHTESTSTSTSTST"
        "SSGSGSGSGSWWWWWGSGSACDEFGHIKACDEFGHIKACDEFGHIKACDEFGHIK"
        "MAMAMAMAMAMACDEFGHIKACDEFGHIKGSGSGSGSTESTTESTTESTACDEFGHIK"
        "APRTEINSEQENCEAAAAAAACDEFGHIK"
    ),
}


def clean_sequence(seq: str) -> str:
    """Return ``seq`` upper-cased with whitespace and a trailing stop ``*`` removed.

    Raises
    ------
    ValueError
        If the sequence is empty or contains characters that are not IUPAC
        amino acid codes.
    """
    s = "".join(seq.split()).upper().rstrip("*")
    if not s:
        raise ValueError("empty sequence")
    bad = sorted(set(s) - VALID_CODES)
    if bad:
        raise ValueError(f"invalid amino acid code(s): {', '.join(bad)}")
    return s


def iter_fasta(path: str | Path) -> Iterator[tuple[str, str]]:
    """Yield ``(header, sequence)`` pairs from a FASTA file."""
    header, chunks = None, []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if header is not None:
                    yield header, clean_sequence("".join(chunks))
                header, chunks = line[1:], []
            else:
                if header is None:
                    raise ValueError(f"{path}: sequence data before first '>' header")
                chunks.append(line)
    if header is not None:
        yield header, clean_sequence("".join(chunks))


def read_fasta(path: str | Path) -> list[tuple[str, str]]:
    """Read every record of a FASTA file into a list of ``(header, sequence)``."""
    return list(iter_fasta(path))


def accession(header: str) -> str:
    """Extract the accession from a UniProt-style header (``sp|P42858|HD_HUMAN ...``)."""
    first = header.split()[0]
    parts = first.split("|")
    return parts[1] if len(parts) >= 3 else first
