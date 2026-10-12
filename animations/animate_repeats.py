"""Animate greedy repeat tiling: each frame adds the next accepted repeat.

Usage:
    python animations/animate_repeats.py [SEQUENCE_OR_FASTA] [-o out.mp4] [--approximate]

Writes MP4 when ffmpeg is available, otherwise GIF (Pillow).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.animation import FFMpegWriter, PillowWriter, writers  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, Normalize  # noqa: E402

from aarepeats import EXAMPLES, find_all_repeats, read_fasta  # noqa: E402
from aarepeats.plotting import INK, INK2, MUTED, use_style  # noqa: E402
from aarepeats.sequences import clean_sequence  # noqa: E402


def animate_repeats(seq: str, repeats, out: Path, fps: int = 2, dpi: int = 150) -> Path:
    """Render one frame per accepted repeat, coloured by word length."""
    use_style()
    cmap = LinearSegmentedColormap.from_list("len", ["#86b6ef", "#2a78d6", "#0d366b"])
    norm = Normalize(vmin=min((r.length for r in repeats), default=2),
                     vmax=max((r.length for r in repeats), default=10))
    if writers.is_available("ffmpeg") and out.suffix == ".mp4":
        writer = FFMpegWriter(fps=fps, bitrate=4000)
    else:
        out = out.with_suffix(".gif")
        writer = PillowWriter(fps=fps)

    fig, ax = plt.subplots(figsize=(12, 2.6))
    with writer.saving(fig, str(out), dpi=dpi):
        for k in range(len(repeats) + 1):
            ax.clear()
            ax.set_xlim(-1, len(seq) + 1)
            ax.set_ylim(0, 1.5)
            ax.set_yticks([])
            ax.grid(False)
            ax.spines["left"].set_visible(False)
            ax.set_xlabel("Residue position")
            ax.hlines(0.5, 0, len(seq), color=MUTED, lw=3)
            for rep in repeats[:k]:
                for pos in rep.occurrences:
                    ax.add_patch(plt.Rectangle((pos, 0.25), rep.length, 0.5,
                                               color=cmap(norm(rep.length)), ec="white", lw=1))
            if k == 0:
                text = f"{len(seq)} residues; {len(repeats)} repeats to place"
            else:
                rep = repeats[k - 1]
                pos = ", ".join(str(p + 1) for p in rep.occurrences)
                text = f"#{k}: '{rep.word}' x{len(rep.occurrences)} at {pos}"
            ax.set_title("Greedy repeat tiling", color=INK)
            ax.text(0, 1.1, text, color=INK2, fontsize=10, fontfamily="monospace")
            writer.grab_frame()
        for _ in range(fps * 2):  # hold the final frame
            writer.grab_frame()
    plt.close(fig)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("input", nargs="?", default="toy_long",
                        help="sequence, FASTA file or example name (default: toy_long)")
    parser.add_argument("-o", "--output", type=Path, default=Path("repeats.mp4"))
    parser.add_argument("--min-length", type=int, default=2)
    parser.add_argument("--min-occurrences", type=int, default=3)
    parser.add_argument("--approximate", action="store_true", help="BLOSUM62 approximate matching")
    args = parser.parse_args()

    if args.input in EXAMPLES:
        seq = EXAMPLES[args.input]
    elif Path(args.input).exists():
        seq = read_fasta(args.input)[0][1]
    else:
        seq = clean_sequence(args.input)

    repeats, seconds = find_all_repeats(seq, args.min_length, args.min_occurrences,
                                        approximate=args.approximate)
    for i, r in enumerate(repeats, 1):
        print(f"{i:>3}. {r.word:<14} x{len(r.occurrences):<3} at {[p + 1 for p in r.occurrences]}")
    print(f"found {len(repeats)} repeats in {seconds:.3f} s")
    print("animation saved to", animate_repeats(seq, repeats, args.output))


if __name__ == "__main__":
    main()
