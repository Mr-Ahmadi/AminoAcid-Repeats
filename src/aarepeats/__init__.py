"""Detection and analysis of amino acid repeats (AARs) in protein sequences.

The package implements four complementary detection strategies:

* :mod:`aarepeats.entropy`   - low-complexity regions via sliding-window Shannon entropy
* :mod:`aarepeats.fourier`   - periodic (tandem) repeats via spectral analysis
* :mod:`aarepeats.alignment` - internal repeats via Smith-Waterman self-alignment
* :mod:`aarepeats.exact`     - exact and approximate (BLOSUM-scored) repeated words

Shared helpers live in :mod:`aarepeats.sequences`, :mod:`aarepeats.scales`,
:mod:`aarepeats.substitution` and :mod:`aarepeats.tandem`.
"""

from .alignment import SelfAlignment, find_internal_repeats
from .entropy import LCRResult, Region, detect_lcr, shannon_entropy, sliding_entropy
from .exact import Repeat, find_all_repeats
from .fourier import (
    PeriodicRegion,
    detect_periodic_repeats,
    local_periodicity,
    periodic_regions,
    power_spectrum,
)
from .sequences import EXAMPLES, clean_sequence, read_fasta
from .tandem import TandemRepeat, find_tandem_repeats

__version__ = "1.0.0"

__all__ = [
    "EXAMPLES",
    "LCRResult",
    "PeriodicRegion",
    "Region",
    "Repeat",
    "SelfAlignment",
    "TandemRepeat",
    "clean_sequence",
    "detect_lcr",
    "detect_periodic_repeats",
    "find_all_repeats",
    "find_internal_repeats",
    "find_tandem_repeats",
    "local_periodicity",
    "periodic_regions",
    "power_spectrum",
    "read_fasta",
    "shannon_entropy",
    "sliding_entropy",
]
