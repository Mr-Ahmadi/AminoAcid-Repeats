"""Build and execute the tutorial notebooks in ``notebooks/``.

Usage:  python scripts/build_notebooks.py
"""

from __future__ import annotations

from pathlib import Path

import nbformat
from nbconvert.preprocessors import ExecutePreprocessor

ROOT = Path(__file__).resolve().parents[1]
NB = ROOT / "notebooks"

SETUP = """\
import matplotlib.pyplot as plt
from aarepeats import EXAMPLES, read_fasta
from aarepeats import plotting as P

P.use_style()
DATA = "../data"
seq = EXAMPLES["toy"]
print(len(seq), seq)"""

NOTEBOOKS = {
    "01_shannon_entropy.ipynb": [
        ("md", """# 1 · Low-complexity regions with Shannon entropy

The entropy of a window's residue composition, $H = -\\sum_a f_a \\log_2 f_a$, is 0 for a homopolymer
and at most $\\log_2 \\min(W, 20)$ bits. Following SEG (Wootton & Federhen, 1993), a window with
$H \\le K_1$ (*trigger*) seeds a low-complexity region (LCR), which grows through neighbouring
windows with $H \\le K_2$ (*extension*)."""),
        ("code", SETUP),
        ("code", """from aarepeats import detect_lcr

res = detect_lcr(seq, window=8, trigger=1.5, extension=2.0)
for r in res.regions:
    print(f"{r.start + 1:>3}-{r.end:<3} H={r.entropy:.2f}  {r.sequence}")
P.plot_entropy(res, seq, 1.5, 2.0)
plt.show()"""),
        ("md", "## Real protein: polyQ and polyG tracts of the androgen receptor (SEG defaults W=12, 2.2/2.5 bits)"),
        ("code", """header, ar = read_fasta(f"{DATA}/P10275.fasta")[0]
res = detect_lcr(ar)
for r in res.regions:
    print(f"{r.start + 1:>4}-{r.end:<4} H={r.entropy:.2f}  {r.sequence[:45]}")
P.plot_entropy(res, ar, 2.2, 2.5)
plt.show()"""),
    ],
    "02_fourier_transform.ipynb": [
        ("md", """# 2 · Periodic repeats with the Fourier transform

The sequence is encoded as 20 indicator channels. For each window and period $p$, the excess score
$E_p \\in [0, 1]$ measures the fraction of the window's variance explained by period-$p$ structure
($E_p = 1$ for a perfect repeat). The spectrum proposes periods; string matching verifies them."""),
        ("code", SETUP),
        ("code", """from aarepeats import detect_periodic_repeats, local_periodicity

for r in detect_periodic_repeats(seq, window=8):
    print(f"{r.start + 1:>3}-{r.end:<3} period {r.period}  ({r.unit}) x {r.copies:.1f}")
P.plot_periodicity_map(local_periodicity(seq, 2, 8, window=8))
plt.show()"""),
        ("md", "## Encoding matters: hydropathy barely separates M/A or S/T (relative power at period 2)"),
        ("code", """from aarepeats import power_spectrum
from aarepeats.scales import AA_SCALES

for name, scale in [("indicator", None), ("Kyte-Doolittle", AA_SCALES["Kyte-Doolittle hydropathy"])]:
    freq, power = power_spectrum(seq, scale)
    print(f"{name:>15}: power at period 2 = {power[-1]:.2f}")"""),
        ("md", "## Collagen: Gly-X-Y periodicity without identical units"),
        ("code", """from aarepeats import periodic_regions

_, col = read_fasta(f"{DATA}/P02452.fasta")[0]
regions = periodic_regions(col)
print([(r.start + 1, r.end, r.period, round(r.mean_score, 2)) for r in regions])
fig, (a, b) = plt.subplots(1, 2, figsize=(10, 3), gridspec_kw=dict(width_ratios=[1, 2.3]))
P.plot_spectrum(*power_spectrum(col), ax=a)
P.plot_periodicity_map(local_periodicity(col, 2, 12, window=36), ax=b, regions=regions)
plt.show()"""),
    ],
    "03_self_alignment.ipynb": [
        ("md", """# 3 · Internal repeats by self-alignment

Smith–Waterman alignment of the sequence against itself (BLOSUM62, linear gap −8, upper triangle),
with Waterman–Eggert declumping so that each reported alignment is distinct. Significance is a
z-score against shuffled sequences. Low-complexity regions dominate unless masked."""),
        ("code", SETUP),
        ("code", """from aarepeats import find_internal_repeats

toy = EXAMPLES["toy_long"]
for masked in (False, True):
    print("LCR masking:", masked)
    for a in find_internal_repeats(toy, min_score=40, max_alignments=4, mask_low_complexity=masked, n_shuffles=20):
        print(f"  {a.start1 + 1}-{a.end1} ~ {a.start2 + 1}-{a.end2}  score {a.score:.0f}  z={a.zscore:.1f}  {a.kind}")
        print(f"    {a.aligned1[:60]}\\n    {a.aligned2[:60]}")"""),
        ("md", "## Sup35: diverged oligopeptide repeats (PQGGYQQYN)"),
        ("code", """_, sup = read_fasta(f"{DATA}/P05453.fasta")[0]
al = find_internal_repeats(sup[:125], min_score=30, max_alignments=3, n_shuffles=50)
for a in al:
    print(f"{a.start1 + 1}-{a.end1} ~ {a.start2 + 1}-{a.end2}  offset {a.offset}  identity {a.identity:.0%}  z={a.zscore:.1f}")
P.plot_dotplot(sup[:125], al)
plt.show()"""),
    ],
    "04_tandem_and_words.ipynb": [
        ("md", """# 4 · Tandem repeats and repeated words

The tandem scanner compares each residue with the one `p` positions downstream; runs of matches are
`p`-periodic stretches. The word finder enumerates (possibly dispersed) repeated words, exactly or
with BLOSUM62 similarity, and tiles them greedily."""),
        ("code", SETUP),
        ("code", """from aarepeats import find_tandem_repeats

for acc in ["P42858", "P10275", "P05453"]:
    header, s = read_fasta(f"{DATA}/{acc}.fasta")[0]
    print(header.split(" OS=")[0])
    for r in find_tandem_repeats(s, min_length=8, max_gap=1):
        print(f"  {r.start + 1:>5}-{r.end:<5} ({r.unit}) x {r.copies:.1f}  identity {r.identity:.0%}")"""),
        ("code", """from aarepeats import find_all_repeats

toy = EXAMPLES["toy_long"]
for approximate in (False, True):
    reps, t = find_all_repeats(toy, min_length=2, min_occurrences=3, approximate=approximate)
    print("approximate" if approximate else "exact", f"({t * 1000:.1f} ms)")
    for r in reps[:4]:
        print(f"  {r.word:<10} x{len(r.occurrences)}  {sorted(set(r.matches))}")"""),
    ],
}


def build() -> None:
    NB.mkdir(exist_ok=True)
    ep = ExecutePreprocessor(timeout=600, kernel_name="python3")
    for name, cells in NOTEBOOKS.items():
        nb = nbformat.v4.new_notebook()
        nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
        nb.cells = [
            nbformat.v4.new_markdown_cell(src) if kind == "md" else nbformat.v4.new_code_cell(src)
            for kind, src in cells
        ]
        ep.preprocess(nb, {"metadata": {"path": str(NB)}})
        nbformat.write(nb, NB / name)
        print("wrote", f"notebooks/{name}")


if __name__ == "__main__":
    build()
