# Amino Acid Repeats: Analysis & Implications

Detection and analysis of amino acid repeats (AARs) in protein sequences. The project contains a
tested Python package, a LaTeX report and a LaTeX (Beamer) presentation. Every figure and number in
the report and slides is regenerated from code.

| Deliverable | Source | Output |
| --- | --- | --- |
| Report (14 pages) | [`report/report.tex`](report/report.tex) | [`report/report.pdf`](report/report.pdf) |
| Presentation (16:9) | [`Presentation/slides.tex`](Presentation/slides.tex) | [`Presentation/slides.pdf`](Presentation/slides.pdf) |
| Python package | [`src/aarepeats/`](src/aarepeats) | `pip install -e .` |

## Methods

| Module | Strategy | Finds |
| --- | --- | --- |
| `aarepeats.entropy` | Sliding-window Shannon entropy with SEG trigger/extension thresholds | Homopolymers, low-complexity regions |
| `aarepeats.tandem` | Shifted self-match `s[i] == s[i+p]` with consensus-aware gap bridging | Exact/near-exact tandem units, copy number |
| `aarepeats.fourier` | Local periodicity score `E_p` (fraction of variance at harmonics of `1/p`), indicator encoding | Periodic patterns, incl. those without identical units (collagen Gly-X-Y) |
| `aarepeats.alignment` | Vectorised Smith–Waterman self-alignment, Waterman–Eggert declumping, LCR masking, shuffle z-scores | Diverged tandem and distributed repeats |
| `aarepeats.exact` | Apriori-style repeated-word growth, exact or BLOSUM62-approximate, greedy tiling | Dispersed repeated motifs |

## Quick start

```bash
make install            # creates .venv and installs the package with dev extras
make test               # 21 unit tests

.venv/bin/aarepeats data/P10275.fasta                 # all methods, text output
.venv/bin/aarepeats MAMAMAMAPRTEINQQQQQQQQ -m tandem   # raw sequence, one method
.venv/bin/aarepeats toy --json                         # built-in example, JSON
```

```python
from aarepeats import read_fasta, detect_lcr, find_tandem_repeats, periodic_regions

_, seq = read_fasta("data/P42858.fasta")[0]          # huntingtin
for r in find_tandem_repeats(seq, min_length=8, max_gap=1):
    print(r.start + 1, r.end, r.unit, r.copies)       # 18 38 Q 21.0 / 39 49 P 11.0 ...
```

## Reproducing the report and slides

```bash
make figures     # report/figures/*.pdf, report/generated/*.tex, results/results.json
make report      # report/report.pdf  (pdflatex + biber)
make slides      # Presentation/slides.pdf (metropolis theme)
make notebooks   # re-execute the tutorial notebooks
```

Requires Python ≥ 3.10 and a TeX Live distribution with `biber`, `metropolis` and `fira`.

## Repository layout

```text
src/aarepeats/      package: entropy, tandem, fourier, alignment, exact, plotting, cli
tests/              unit tests (known-answer cases for every detector)
data/               UniProtKB sequences: HTT P42858, AR P10275, COL1A1 P02452, SUP35 P05453
scripts/            make_figures.py (all report artefacts), build_notebooks.py
notebooks/          executed tutorials, one per method
animations/         animate_repeats.py: MP4/GIF of greedy repeat tiling
report/             LaTeX report, bibliography, generated figures/tables
Presentation/       LaTeX Beamer slides
results/            machine-readable results for the four proteins
```

## Key results

* **Huntingtin**: Q₂₁ (residues 18–38) followed by P₁₁ (39–49), reported as two separate tracts.
* **Androgen receptor**: Q₂₃ (58–80) and G₂₃ (451–473), the two deepest entropy minima.
* **Collagen α1(I)**: a single period-3 spectral peak about 100× the mean power, localised to residues 187–1183. Only Fourier analysis reveals this pattern.
* **Sup35**: diverged oligopeptide repeats recovered by self-alignment (z ≈ 6.5).
* **Low-complexity masking matters**: unmasked, the AR polyQ/polyP/polyG tracts chain into one spurious 430-residue "repeat".

See the report's appendix for the defects fixed relative to the original prototype (int64 rolling-hash
overflow, a mislabelled polarity scale, redundant alignments, among others).

## Author

Ali Ahmadi Esfidi, Feb, 2025
