"""Regenerate every figure, table and number used in the report and slides.

Usage:  python scripts/make_figures.py [--quick]

Outputs
    report/figures/*.pdf      figures
    report/generated/*.tex    LaTeX tables and \\newcommand macros
    results/results.json      machine-readable results for all proteins
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path

import matplotlib
import matplotlib.ticker

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from aarepeats import (  # noqa: E402
    EXAMPLES,
    detect_lcr,
    detect_periodic_repeats,
    find_all_repeats,
    find_internal_repeats,
    find_tandem_repeats,
    local_periodicity,
    periodic_regions,
    power_spectrum,
)
from aarepeats import plotting as P  # noqa: E402
from aarepeats.sequences import accession, read_fasta  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "report" / "figures"
GEN = ROOT / "report" / "generated"
RES = ROOT / "results"

PROTEINS = [  # accession, short name, display name, organism
    ("P42858", "HTT", "Huntingtin", "\\textit{H. sapiens}"),
    ("P10275", "AR", "Androgen receptor", "\\textit{H. sapiens}"),
    ("P02452", "COL1A1", "Collagen $\\alpha$1(I) chain", "\\textit{H. sapiens}"),
    ("P05453", "SUP35", "Eukaryotic release factor 3 (Sup35)", "\\textit{S. cerevisiae}"),
]

# Parameters used throughout (also printed into the report).
TANDEM = dict(min_length=8, max_gap=1, min_identity=0.8)
ALIGN = dict(min_score=35, max_alignments=6, mask_low_complexity=True, n_shuffles=50, seed=1)
Z_SIGNIFICANT = 4.0


def load(acc: str) -> tuple[str, str]:
    header, seq = read_fasta(ROOT / "data" / f"{acc}.fasta")[0]
    assert accession(header) == acc
    return header, seq


def tex_escape(s: str) -> str:
    return s.replace("_", "\\_").replace("%", "\\%").replace("&", "\\&")


def save(fig, name: str) -> None:
    fig.savefig(FIG / f"{name}.pdf")
    plt.close(fig)
    print("  wrote", f"figures/{name}.pdf")


# --------------------------------------------------------------------------- analyses

def analyse_protein(seq: str, n_shuffles: int) -> dict:
    lcr = detect_lcr(seq)
    tandem = find_tandem_repeats(seq, **TANDEM)
    fourier = detect_periodic_repeats(seq)
    spectral = periodic_regions(seq)
    align = find_internal_repeats(seq, **{**ALIGN, "n_shuffles": n_shuffles})
    words, _ = find_all_repeats(seq, min_length=4, min_occurrences=3)
    return dict(lcr=lcr, tandem=tandem, fourier=fourier, spectral=spectral, align=align, words=words)


def ground_truth_table(macros: dict) -> None:
    """Which method recovers each designed element of the synthetic sequence?"""
    seq = EXAMPLES["toy"]
    elements = [  # name, start, end (0-based half-open), class
        ("(MA)$_6$", 0, 12, "perfect tandem"),
        ("(A)$_6$", 25, 31, "homopolymer"),
        ("(H)$_4$", 31, 35, "homopolymer"),
        ("(ST)$_{5}$S", 37, 48, "perfect tandem"),
        ("(SG)$_{4}$S", 48, 57, "perfect tandem"),
        ("(W)$_5$", 57, 62, "homopolymer"),
        ("PRTEINSEQENCE", 12, 25, "non-repeat spacer"),
    ]
    lcr = [(r.start, r.end) for r in detect_lcr(seq, window=8, trigger=1.5, extension=2.0).regions]
    tan = [(r.start, r.end) for r in find_tandem_repeats(seq, min_length=4)]
    fou = [(r.start, r.end) for r in detect_periodic_repeats(seq, window=8)]
    ali = []
    for a in find_internal_repeats(seq, min_score=15, max_alignments=10):
        ali += [(a.start1, a.end1), (a.start2, a.end2)]
    words, _ = find_all_repeats(seq, min_length=2, min_occurrences=3)
    wrd = [(p, p + w.length) for w in words for p in w.occurrences]
    methods = [("Entropy", lcr), ("Tandem", tan), ("Fourier", fou), ("Self-align.", ali), ("Words", wrd)]

    def hit(intervals, a, b):
        covered = np.zeros(len(seq), bool)
        for s, e in intervals:
            covered[s:e] = True
        return covered[a:b].mean() >= 0.5

    rows = []
    for name, a, b, kind in elements:
        marks = ["\\cmark" if hit(iv, a, b) else "\\xmark" for _, iv in methods]
        rows.append(f"{name} & {a + 1}--{b} & {kind} & " + " & ".join(marks) + " \\\\")
    header = " & ".join(m for m, _ in methods)
    (GEN / "toy_truth.tex").write_text(
        "\\begin{tabular}{@{}lllccccc@{}}\n\\toprule\n"
        f"Element & Residues & Class & {header} \\\\\n\\midrule\n"
        + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n"
    )
    macros["toyLength"] = len(seq)
    print("  wrote generated/toy_truth.tex")


# --------------------------------------------------------------------------- figures

def fig_entropy(seqs: dict, macros: dict) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(7, 4.1), gridspec_kw=dict(hspace=0.55))
    htt = seqs["HTT"][:110]
    res = detect_lcr(htt)
    P.plot_entropy(res, htt, 2.2, 2.5, ax=axes[0], show_residues=False)
    axes[0].set_title("(a) Huntingtin, residues 1–110 (W = 12)")
    for r in res.regions:
        axes[0].text(r.start + 13, 0.12, f"LCR {r.start + 1}–{r.end}", ha="left",
                     va="bottom", fontsize=7.5, color=P.INK)
    ar = seqs["AR"]
    res = detect_lcr(ar)
    P.plot_entropy(res, ar, 2.2, 2.5, ax=axes[1])
    axes[1].set_title("(b) Androgen receptor, full length")
    for label, (a, b) in {"polyQ": (57, 80), "polyG": (450, 473)}.items():
        axes[1].text(b + 6, 0.25, label, ha="left", fontsize=7.5, color=P.INK)
    save(fig, "entropy")


def fig_toy(macros: dict) -> None:
    seq = EXAMPLES["toy"]
    fig, axes = plt.subplots(2, 1, figsize=(7, 3.9), gridspec_kw=dict(hspace=0.5, height_ratios=[1.1, 1]))
    res = detect_lcr(seq, window=8, trigger=1.5, extension=2.0)
    P.plot_entropy(res, seq, 1.5, 2.0, ax=axes[0])
    axes[0].set_title("(a) Sliding-window entropy (W = 8)")
    pmap = local_periodicity(seq, 2, 8, window=8)
    P.plot_periodicity_map(pmap, ax=axes[1])
    axes[1].set_title("(b) Local periodicity score (indicator encoding, W = 8)")
    axes[1].set_xticks(range(1, len(seq) + 1))
    axes[1].set_xticklabels(list(seq), fontfamily="monospace", fontsize=6.5)
    axes[1].tick_params(axis="x", length=0)
    axes[1].set_xlabel("")
    save(fig, "toy")


def fig_fourier(seqs: dict, macros: dict) -> None:
    col = seqs["COL1A1"]
    freq, power = power_spectrum(col)
    k = int(np.argmax(power))
    macros["colPeakPeriod"] = f"{1 / freq[k]:.2f}"
    macros["colPeakPower"] = f"{power[k]:.0f}"
    second = np.sort(power)[-2]
    macros["colSecondPower"] = f"{second:.0f}"
    regions = periodic_regions(col)
    main = max(regions, key=lambda r: r.length)
    macros["colRegionStart"], macros["colRegionEnd"] = main.start + 1, main.end
    macros["colRegionLen"] = main.length
    macros["colRegionScore"] = f"{main.mean_score:.2f}"
    gly = col[main.start:main.end]
    phase = max(range(3), key=lambda f: gly[f::3].count("G"))
    macros["colGlyFraction"] = f"{100 * gly[phase::3].count('G') / len(gly[phase::3]):.1f}"

    fig = plt.figure(figsize=(7, 2.5))
    gs = fig.add_gridspec(1, 2, width_ratios=[1, 2.3], wspace=0.28)
    ax1 = fig.add_subplot(gs[0])
    P.plot_spectrum(freq, power, ax=ax1, max_period=12)
    ax1.set_title("(a) Global spectrum")
    ax2 = fig.add_subplot(gs[1])
    pmap = local_periodicity(col, 2, 12, window=36)
    P.plot_periodicity_map(pmap, ax=ax2, regions=regions)
    ax2.set_title("(b) Local periodicity, W = 36")
    save(fig, "fourier_collagen")

    # Encoding comparison on the synthetic sequence: global power at period 2.
    from aarepeats.scales import AA_SCALES

    seq = EXAMPLES["toy"]
    _, kd_power = power_spectrum(seq, AA_SCALES["Kyte-Doolittle hydropathy"])
    _, in_power = power_spectrum(seq)
    macros["toyNyquistIndicator"] = f"{in_power[-1]:.1f}"
    macros["toyNyquistKD"] = f"{kd_power[-1]:.1f}"


def fig_dotplots(seqs: dict, macros: dict) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(7, 3.6), gridspec_kw=dict(wspace=0.32))
    sup = seqs["SUP35"][:125]
    al = find_internal_repeats(sup, min_score=30, max_alignments=3, n_shuffles=50, seed=1)
    P.plot_dotplot(sup, al, window=9, threshold=0.66, ax=axes[0])
    axes[0].set_title("(a) Sup35 N-domain, residues 1–125")
    best = al[0]
    macros.update(supAlnScore=f"{best.score:.0f}", supAlnZ=f"{best.zscore:.1f}",
                  supAlnOffset=best.offset, supAlnIdentity=f"{100 * best.identity:.0f}",
                  supAlnA=f"{best.start1 + 1}--{best.end1}", supAlnB=f"{best.start2 + 1}--{best.end2}")

    toy = EXAMPLES["toy_long"]
    al = find_internal_repeats(toy, min_score=40, max_alignments=4, mask_low_complexity=True,
                               n_shuffles=50, seed=1)
    P.plot_dotplot(toy, al, window=9, threshold=0.66, ax=axes[1])
    axes[1].set_title("(b) Synthetic sequence, LCRs masked")
    save(fig, "dotplot")


def fig_repeat_map(seqs: dict, results: dict) -> None:
    fig, axes = plt.subplots(4, 1, figsize=(7, 6.2), gridspec_kw=dict(hspace=0.9))
    notes = {
        "HTT": [(17, 49, "Q$_{21}$P$_{11}$")],
        "AR": [(57, 80, "Q$_{23}$"), (450, 473, "G$_{23}$")],
        "COL1A1": [],
        "SUP35": [],
    }
    for ax, (acc, short, name, _) in zip(axes, PROTEINS):
        r = results[short]
        sig = [a for a in r["align"] if a.zscore is not None and a.zscore >= Z_SIGNIFICANT]
        tracks = {
            "Entropy (LCR)": [(x.start, x.end) for x in r["lcr"].regions],
            "Tandem": [(x.start, x.end) for x in r["tandem"]],
            "Fourier (periodic)": [(x.start, x.end) for x in r["spectral"]]
            + [(x.start, x.end) for x in r["fourier"]],
            "Self-alignment": [iv for a in sig for iv in ((a.start1, a.end1), (a.start2, a.end2))],
        }
        P.plot_repeat_map(len(seqs[short]), tracks, ax=ax, annotations=notes[short],
                          title=f"{name} ({acc}, {len(seqs[short])} aa)")
    axes[-1].set_xlabel("Residue position")
    save(fig, "repeat_map")


def fig_runtime(quick: bool, macros: dict) -> None:
    rng = np.random.default_rng(0)
    lengths = [250, 500, 1000, 2000] if quick else [250, 500, 1000, 2000, 4000]
    methods = {
        "Entropy": lambda s: detect_lcr(s),
        "Tandem": lambda s: find_tandem_repeats(s, **TANDEM),
        "Fourier": lambda s: detect_periodic_repeats(s),
        "Words": lambda s: find_all_repeats(s, min_length=4, min_occurrences=3),
        "Self-alignment": lambda s: find_internal_repeats(s, min_score=35, max_alignments=1),
    }
    times = {m: [] for m in methods}
    for n in lengths:
        s = "".join(rng.choice(list("ACDEFGHIKLMNPQRSTVWY"), size=n))
        for m, f in methods.items():
            best = np.inf
            for _ in range(3 if n <= 1000 else 1):
                t = time.perf_counter()
                f(s)
                best = min(best, time.perf_counter() - t)
            times[m].append(best)
    fig, ax = plt.subplots(figsize=(4.6, 2.6))
    colors = [P.ORANGE, P.BLUE, P.AQUA, P.MUTED, P.YELLOW]
    markers = ["o", "s", "^", "D", "v"]
    for (m, t), c, mk in zip(times.items(), colors, markers):
        ax.loglog(lengths, t, marker=mk, ms=4.5, lw=1.6, color=c, label=m)
    ax.set_xticks(lengths)
    ax.set_xticklabels([f"{n:,}" for n in lengths])
    ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    ax.set_xlabel("Sequence length (residues)")
    ax.set_ylabel("Wall time (s)")
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), fontsize=7.5)
    save(fig, "runtime")
    slope = np.polyfit(np.log(lengths[-3:]), np.log(times["Self-alignment"][-3:]), 1)[0]
    macros["alignSlope"] = f"{slope:.1f}"
    macros["alignTimeMax"] = f"{times['Self-alignment'][-1]:.1f}"
    macros["runtimeMaxLength"] = f"{lengths[-1]:,}".replace(",", "{,}")
    macros["entropyTimeMax"] = f"{1000 * times['Entropy'][-1]:.0f}"


# --------------------------------------------------------------------------- tables

def write_tables(seqs: dict, headers: dict, results: dict, macros: dict) -> None:
    rows = []
    for acc, short, name, org in PROTEINS:
        sv = headers[short].split("SV=")[-1]
        rows.append(f"{name} & {short} & {acc} (v{sv}) & {org} & {len(seqs[short]):,} \\\\".replace(",", "{,}"))
    (GEN / "dataset.tex").write_text(
        "\\begin{tabular}{@{}lllll@{}}\n\\toprule\nProtein & Gene & UniProtKB & Organism & Length \\\\\n\\midrule\n"
        + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n"
    )

    rows = []
    for acc, short, name, _ in PROTEINS:
        reps = sorted(results[short]["tandem"], key=lambda r: -r.length)[:5]
        reps.sort(key=lambda r: r.start)
        for k, r in enumerate(reps):
            unit = r.unit if len(r.unit) <= 10 else r.unit[:10] + "…"
            first = f"\\multirow{{{len(reps)}}}{{*}}{{{short}}}" if k == 0 else ""
            rows.append(
                f"{first} & {r.start + 1}--{r.end} & \\texttt{{{unit}}} & {r.period} & "
                f"{r.copies:.1f} & {100 * r.identity:.0f} \\\\"
            )
        rows.append("\\addlinespace")
    (GEN / "tandem.tex").write_text(
        "\\begin{tabular}{@{}lrlrrr@{}}\n\\toprule\nProtein & Residues & Unit & Period & Copies & Identity (\\%) \\\\\n\\midrule\n"
        + "\n".join(rows[:-1]) + "\n\\bottomrule\n\\end{tabular}\n"
    )

    rows = []
    for acc, short, name, _ in PROTEINS:
        r = results[short]
        n = len(seqs[short])
        cov = np.zeros(n, bool)
        for x in r["lcr"].regions:
            cov[x.start:x.end] = True
        spectral = r["spectral"]
        periods = sorted({x.period for x in spectral})
        if len(periods) == 1:
            spec = ", ".join(f"{x.start + 1}--{x.end}" for x in spectral) + f" ($p{{=}}{periods[0]}$)"
        else:
            spec = ", ".join(f"{x.start + 1}--{x.end} ($p{{=}}{x.period}$)" for x in spectral) or "--"
        sig = [a for a in r["align"] if a.zscore >= Z_SIGNIFICANT]
        top = r["align"][0] if r["align"] else None
        top_s = f"{top.score:.0f} ({top.zscore:.1f})" if top else "--"
        rows.append(
            f"{short} & {len(r['lcr'].regions)} & {100 * cov.mean():.1f} & {len(r['tandem'])} & "
            f"{len(r['fourier'])} & {spec} & {top_s} & {len(sig)} \\\\"
        )
    (GEN / "summary.tex").write_text(
        "\\begin{tabular}{@{}lrrrrlrr@{}}\n\\toprule\n"
        " & \\multicolumn{2}{c}{Entropy} & Tandem & \\multicolumn{2}{c}{Fourier} & \\multicolumn{2}{c}{Self-alignment (masked)} \\\\\n"
        "\\cmidrule(lr){2-3}\\cmidrule(lr){4-4}\\cmidrule(lr){5-6}\\cmidrule(l){7-8}\n"
        "Protein & LCRs & Cover (\\%) & repeats & verified & periodic regions & top score ($z$) & $z \\ge 4$ \\\\\n\\midrule\n"
        + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n"
    )

    # Key numbers quoted in the text.
    htt, ar, sup = results["HTT"], results["AR"], results["SUP35"]
    q = next(r for r in htt["tandem"] if r.unit == "Q")
    p = next(r for r in htt["tandem"] if r.unit == "P")
    macros.update(httQStart=q.start + 1, httQEnd=q.end, httQLen=q.length,
                  httPStart=p.start + 1, httPEnd=p.end, httPLen=p.length)
    lcr0 = htt["lcr"].regions[0]
    macros.update(httLcrStart=lcr0.start + 1, httLcrEnd=lcr0.end)
    q = next(r for r in ar["tandem"] if r.unit == "Q" and r.length > 15)
    g = next(r for r in ar["tandem"] if r.unit == "G")
    macros.update(arQStart=q.start + 1, arQEnd=q.end, arQLen=q.length,
                  arGStart=g.start + 1, arGEnd=g.end, arGLen=g.length)
    s9 = next(r for r in sup["tandem"] if r.period == 9)
    macros.update(supUnit=s9.unit, supUnitStart=s9.start + 1, supUnitEnd=s9.end)
    col = results["COL1A1"]
    if col["align"]:
        a = col["align"][0]
        macros.update(colAlnScore=f"{a.score:.0f}", colAlnZ=f"{a.zscore:.1f}",
                      colAlnIdentity=f"{100 * a.identity:.0f}")
    h = htt["align"][0]
    macros.update(httAlnZ=f"{h.zscore:.1f}")
    macros["tandemMinLength"] = TANDEM["min_length"]
    macros["zSignificant"] = f"{Z_SIGNIFICANT:.0f}"
    macros["nShuffles"] = ALIGN["n_shuffles"]

    # Unmasked AR alignment: low-complexity runs chain into one long "repeat".
    raw = find_internal_repeats(seqs["AR"], min_score=35, max_alignments=1, n_shuffles=20, seed=1)[0]
    macros.update(arRawStart=raw.start1 + 1, arRawEnd=raw.end2, arRawLen=raw.end2 - raw.start1,
                  arRawZ=f"{raw.zscore:.0f}", arRawIdentity=f"{100 * raw.identity:.0f}",
                  arRawOffset=raw.offset)


def write_macros(macros: dict) -> None:
    lines = ["% Generated by scripts/make_figures.py -- do not edit by hand."]
    for k, v in sorted(macros.items()):
        lines.append(f"\\newcommand{{\\{k}}}{{{v}}}")
    (GEN / "macros.tex").write_text("\n".join(lines) + "\n")
    print("  wrote generated/macros.tex")


def to_json(results: dict) -> dict:
    out = {}
    for short, r in results.items():
        out[short] = {
            "lcr": [asdict(x) for x in r["lcr"].regions],
            "tandem": [asdict(x) for x in r["tandem"]],
            "fourier_verified": [asdict(x) for x in r["fourier"]],
            "fourier_regions": [asdict(x) for x in r["spectral"]],
            "self_alignment": [{**asdict(a), "identity": a.identity, "kind": a.kind} for a in r["align"]],
            "words": [{"word": w.word, "occurrences": w.occurrences} for w in r["words"][:20]],
        }
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true", help="fewer shuffles and benchmark sizes")
    args = parser.parse_args()
    for d in (FIG, GEN, RES):
        d.mkdir(parents=True, exist_ok=True)
    P.use_style()

    headers, seqs = {}, {}
    for acc, short, _, _ in PROTEINS:
        headers[short], seqs[short] = load(acc)

    print("Analysing proteins ...")
    n_shuffles = 10 if args.quick else ALIGN["n_shuffles"]
    results = {short: analyse_protein(seqs[short], n_shuffles) for _, short, _, _ in PROTEINS}
    (RES / "results.json").write_text(json.dumps(to_json(results), indent=1))

    macros: dict = {}
    print("Figures ...")
    fig_entropy(seqs, macros)
    fig_toy(macros)
    fig_fourier(seqs, macros)
    fig_dotplots(seqs, macros)
    fig_repeat_map(seqs, results)
    fig_runtime(args.quick, macros)
    print("Tables ...")
    ground_truth_table(macros)
    write_tables(seqs, headers, results, macros)
    write_macros(macros)


if __name__ == "__main__":
    main()
