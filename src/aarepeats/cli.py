"""Command-line interface: ``aarepeats SEQUENCE_OR_FASTA [--method ...] [--json]``."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

from .alignment import find_internal_repeats
from .entropy import detect_lcr
from .exact import find_all_repeats
from .fourier import detect_periodic_repeats, periodic_regions
from .sequences import EXAMPLES, clean_sequence, read_fasta
from .tandem import find_tandem_repeats

METHODS = ("entropy", "tandem", "fourier", "align", "exact")


def _load(arg: str) -> list[tuple[str, str]]:
    if arg in EXAMPLES:
        return [(arg, EXAMPLES[arg])]
    path = Path(arg)
    if path.exists():
        return read_fasta(path)
    return [("input", clean_sequence(arg))]


def analyse(seq: str, methods: tuple[str, ...] = METHODS) -> dict:
    """Run the selected detectors and return JSON-serialisable results (0-based, half-open)."""
    out: dict = {"length": len(seq)}
    if "entropy" in methods:
        out["entropy"] = [
            {k: v for k, v in asdict(r).items()} for r in detect_lcr(seq).regions
        ]
    if "tandem" in methods:
        out["tandem"] = [asdict(r) for r in find_tandem_repeats(seq, max_gap=1)]
    if "fourier" in methods:
        out["fourier"] = {
            "verified_repeats": [asdict(r) for r in detect_periodic_repeats(seq)],
            "periodic_regions": [asdict(r) for r in periodic_regions(seq)],
        }
    if "align" in methods:
        out["align"] = [
            {**asdict(a), "kind": a.kind, "identity": a.identity}
            for a in find_internal_repeats(seq, mask_low_complexity=True, n_shuffles=20)
        ]
    if "exact" in methods:
        repeats, _ = find_all_repeats(seq, min_length=3, min_occurrences=2)
        out["exact"] = [
            {"word": r.word, "occurrences": r.occurrences} for r in repeats if r.length >= 4
        ]
    return out


def _print_text(name: str, res: dict) -> None:
    print(f"# {name}  ({res['length']} aa; positions are 1-based, inclusive)")
    for r in res.get("entropy", []):
        print(f"  LCR      {r['start'] + 1:>6}-{r['end']:<6} H={r['entropy']:.2f}  {r['sequence'][:40]}")
    for r in res.get("tandem", []):
        print(
            f"  tandem   {r['start'] + 1:>6}-{r['end']:<6} ({r['unit']})x{(r['end'] - r['start']) / r['period']:.1f}"
            f"  id={r['identity']:.2f}"
        )
    if "fourier" in res:
        for r in res["fourier"]["verified_repeats"]:
            print(f"  fourier  {r['start'] + 1:>6}-{r['end']:<6} period {r['period']} ({r['unit']})")
        for r in res["fourier"]["periodic_regions"]:
            print(f"  spectral {r['start'] + 1:>6}-{r['end']:<6} period {r['period']}  E={r['mean_score']:.2f}")
    for a in res.get("align", []):
        z = f"z={a['zscore']:.1f}" if a["zscore"] is not None else ""
        print(
            f"  align    {a['start1'] + 1}-{a['end1']} ~ {a['start2'] + 1}-{a['end2']}"
            f"  score={a['score']:.0f} {z} id={a['identity']:.2f} {a['kind']}"
        )
    for r in res.get("exact", [])[:15]:
        pos = ",".join(str(p + 1) for p in r["occurrences"])
        print(f"  word     {r['word']:<12} at {pos}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="aarepeats", description="Detect amino acid repeats in protein sequences."
    )
    parser.add_argument("input", help="FASTA file, raw sequence, or example name (" + ", ".join(EXAMPLES) + ")")
    parser.add_argument(
        "-m", "--method", action="append", choices=METHODS,
        help="detector to run (repeatable; default: all)",
    )
    parser.add_argument("--json", action="store_true", help="emit JSON instead of text")
    args = parser.parse_args(argv)

    try:
        records = _load(args.input)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    methods = tuple(args.method) if args.method else METHODS
    results = {name: analyse(seq, methods) for name, seq in records}
    if args.json:
        json.dump(results, sys.stdout, indent=2)
        print()
    else:
        for name, res in results.items():
            _print_text(name, res)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
