"""Physico-chemical amino acid scales used to turn sequences into numeric signals."""

from __future__ import annotations

#: Each scale maps the 20 standard amino acids to a value.
#: Sources: Kyte & Doolittle (1982); Grantham (1974); Zimmerman et al. (1968);
#: Hopp & Woods (1981).
AA_SCALES: dict[str, dict[str, float]] = {
    "Kyte-Doolittle hydropathy": {
        "A": 1.8, "R": -4.5, "N": -3.5, "D": -3.5, "C": 2.5,
        "Q": -3.5, "E": -3.5, "G": -0.4, "H": -3.2, "I": 4.5,
        "L": 3.8, "K": -3.9, "M": 1.9, "F": 2.8, "P": -1.6,
        "S": -0.8, "T": -0.7, "W": -0.9, "Y": -1.3, "V": 4.2,
    },
    # Previously mislabelled "Zimmerman Polarity": these values are Grantham's
    # polarity scale. Zimmerman's polarity values range from 0 to 52.
    "Grantham polarity": {
        "A": 8.1, "R": 10.5, "N": 11.6, "D": 13.0, "C": 5.5,
        "Q": 10.5, "E": 12.3, "G": 9.0, "H": 10.4, "I": 5.2,
        "L": 4.9, "K": 11.3, "M": 5.7, "F": 5.2, "P": 8.0,
        "S": 9.2, "T": 8.6, "W": 5.4, "Y": 6.2, "V": 5.9,
    },
    "Zimmerman bulkiness": {
        "A": 11.5, "R": 14.28, "N": 12.82, "D": 11.68, "C": 13.46,
        "Q": 14.45, "E": 13.57, "G": 3.4, "H": 13.69, "I": 21.40,
        "L": 21.40, "K": 15.71, "M": 16.25, "F": 19.80, "P": 17.43,
        "S": 9.47, "T": 15.77, "W": 21.67, "Y": 18.03, "V": 21.57,
    },
    "Hopp-Woods hydrophilicity": {
        "A": -0.5, "R": 3.0, "N": 0.2, "D": 3.0, "C": -1.0,
        "Q": 0.2, "E": 3.0, "G": 0.0, "H": -0.5, "I": -1.8,
        "L": -1.8, "K": 3.0, "M": -1.3, "F": -2.5, "P": 0.0,
        "S": 0.3, "T": -0.4, "W": -3.4, "Y": -2.3, "V": -1.5,
    },
}


def get_scale(name: str) -> dict[str, float]:
    """Look up a scale by name, case-insensitively."""
    for key, scale in AA_SCALES.items():
        if key.lower() == name.lower():
            return scale
    raise KeyError(f"unknown scale {name!r}; choose from {list(AA_SCALES)}")
