import json
import math

import numpy as np
import pytest

from aarepeats import (
    EXAMPLES,
    clean_sequence,
    detect_lcr,
    detect_periodic_repeats,
    find_all_repeats,
    find_internal_repeats,
    find_tandem_repeats,
    local_periodicity,
    periodic_regions,
    power_spectrum,
    read_fasta,
    shannon_entropy,
    sliding_entropy,
)
from aarepeats.alignment import windowed_dotplot
from aarepeats.cli import main
from aarepeats.exact import repeated_words
from aarepeats.scales import AA_SCALES, get_scale

TOY = EXAMPLES["toy"]


# --- sequences ---------------------------------------------------------------

def test_clean_sequence_normalises_and_validates():
    assert clean_sequence(" ma ma\nQ* ") == "MAMAQ"
    with pytest.raises(ValueError):
        clean_sequence("MA1")
    with pytest.raises(ValueError):
        clean_sequence("   ")


def test_read_fasta(tmp_path):
    path = tmp_path / "x.fasta"
    path.write_text(">sp|P1|A one\nMAMA\nQQ\n>two\nGGG\n")
    assert read_fasta(path) == [("sp|P1|A one", "MAMAQQ"), ("two", "GGG")]


def test_scales_cover_standard_alphabet():
    for scale in AA_SCALES.values():
        assert set(scale) == set("ACDEFGHIKLMNPQRSTVWY")
    assert get_scale("grantham POLARITY")["D"] == 13.0


# --- entropy -----------------------------------------------------------------

def test_shannon_entropy_known_values():
    assert shannon_entropy("AAAA") == 0
    assert shannon_entropy("ABAB") == pytest.approx(1.0)
    assert shannon_entropy("ACDEFGHIKLMNPQRSTVWY") == pytest.approx(math.log2(20))


def test_sliding_entropy_matches_direct_computation():
    w = 7
    fast = sliding_entropy(TOY, w)
    slow = [shannon_entropy(TOY[i : i + w]) for i in range(len(TOY) - w + 1)]
    np.testing.assert_allclose(fast, slow, atol=1e-12)


def test_detect_lcr_finds_polyq_in_complex_background():
    flank = "MKTWLVDEFNRHICPY" * 2
    seq = flank + "Q" * 20 + flank
    regions = detect_lcr(seq).regions
    assert len(regions) == 1
    r = regions[0]
    assert r.start <= 32 and r.end >= 52  # covers the Q run
    assert r.length <= 20 + 2 * 11  # at most window - 1 overhang on each side
    assert detect_lcr(flank * 3).regions == []


# --- tandem ------------------------------------------------------------------

def test_tandem_reports_primitive_units():
    reps = find_tandem_repeats(TOY)
    got = [(r.start, r.end, r.unit) for r in reps]
    assert got == [(0, 12, "MA"), (25, 31, "A"), (37, 48, "ST"), (48, 57, "SG")]


def test_tandem_does_not_merge_adjacent_homopolymers():
    reps = find_tandem_repeats("MKT" + "Q" * 21 + "P" * 11 + "WKT", max_gap=1)
    assert [(r.unit, r.length) for r in reps] == [("Q", 21), ("P", 11)]


def test_tandem_bridges_single_substitution():
    reps = find_tandem_repeats("QQQQQHQQQQQQ", max_gap=1)
    assert len(reps) == 1 and reps[0].unit == "Q" and reps[0].length == 12
    assert reps[0].identity == pytest.approx(11 / 12)
    assert all(r.length < 12 for r in find_tandem_repeats("QQQQQHQQQQQQ"))  # exact mode


# --- fourier -----------------------------------------------------------------

def test_periodicity_score_is_one_for_perfect_repeat():
    m = local_periodicity("GPP" * 20, min_period=2, max_period=6, window=12)
    assert m.score[m.periods == 3].min() == pytest.approx(1.0)
    assert m.score[m.periods == 6].min() == pytest.approx(1.0)  # multiple of 3
    assert m.score[m.periods == 2].max() < 0.1
    assert set(m.best_period(0.5)) == {3}


def test_power_spectrum_peak_at_period():
    freq, power = power_spectrum("GAP" * 40)
    assert 1 / freq[np.argmax(power)] == pytest.approx(3.0)


def test_fourier_detects_tandem_repeats_hydropathy_cannot_separate():
    reps = detect_periodic_repeats(TOY, window=8)
    assert {(r.start, r.end, r.unit) for r in reps} >= {(0, 12, "MA"), (37, 48, "ST")}


def test_periodic_regions_find_collagen_like_domain():
    rng = np.random.default_rng(1)
    xy = "".join(rng.choice(list("PAEKRSLQD"), size=2 * 100))
    helix = "".join("G" + xy[2 * i : 2 * i + 2] for i in range(100))
    flank = "".join(rng.choice(list("ACDEFHIKLMNQRSTVWY"), size=80))
    regions = periodic_regions(flank + helix + flank)
    assert len(regions) == 1
    r = regions[0]
    assert r.period == 3
    assert abs(r.start - 80) < 20 and abs(r.end - 380) < 20


# --- alignment ---------------------------------------------------------------

def test_self_alignment_finds_dispersed_copy():
    rng = np.random.default_rng(0)
    background = "".join(rng.choice(list("ACDEFGHIKLMNPQRSTVWY"), size=120))
    motif = "WCHYKRPDEF"
    seq = background[:30] + motif + background[30:90] + motif + background[90:]
    best = find_internal_repeats(seq, min_score=40, max_alignments=1, n_shuffles=10)[0]
    # Local alignment may extend a few residues into flanks with positive scores.
    assert best.start1 <= 30 and best.end1 >= 40 and best.end1 - best.start1 <= 16
    assert best.start2 <= 100 and best.end2 >= 110 and best.offset == 70
    assert best.kind == "dispersed" and best.identity >= 0.8
    assert best.zscore > 3


def test_self_alignment_results_are_not_redundant():
    found = find_internal_repeats(EXAMPLES["toy_long"], min_score=30, max_alignments=10)
    assert len(found) >= 3
    for i, a in enumerate(found):
        for b in found[:i]:
            assert not ((a.start1, a.end1, a.start2, a.end2) == (b.start1, b.end1, b.start2, b.end2))
    scores = [a.score for a in found]
    assert scores == sorted(scores, reverse=True)


def test_windowed_dotplot_diagonal():
    d = windowed_dotplot("ACDEFGHIK", window=3)
    assert d[2, 2] == 3 and d[8, 8] == 3 and d[3, 2] < 3


# --- exact -------------------------------------------------------------------

def test_repeated_words_non_overlapping_counts():
    words = repeated_words("AAAAAA", min_length=2, min_occurrences=2)
    assert words["AA"] == [0, 2, 4]
    assert words["AAA"] == [0, 3]
    assert "AAAA" not in words


def test_exact_repeats_long_word_without_hash_collisions():
    # Words longer than 8 residues broke the original int64 rolling hash.
    unit = "ACDEFGHIKLMNPQRSTVWY"
    seq = unit + "GGGG" + unit + "WWWW" + unit
    repeats, _ = find_all_repeats(seq, min_length=2, min_occurrences=3)
    assert repeats[0].word == unit and repeats[0].occurrences == [0, 24, 48]


def test_approximate_repeats_accept_conservative_substitutions():
    seq = "ACDEFGHIK" + "WW" + "ACDEFGHIK" + "WW" + "SCDEFGHIK"  # A->S scores +1 in BLOSUM62
    exact, _ = find_all_repeats(seq, min_length=9, min_occurrences=3)
    approx, _ = find_all_repeats(seq, min_length=9, min_occurrences=2, approximate=True, similarity=0.85)
    assert exact == []
    assert approx[0].occurrences == [0, 11, 22]


# --- cli ---------------------------------------------------------------------

def test_cli_json(capsys):
    assert main(["toy", "-m", "tandem", "-m", "entropy", "--json"]) == 0
    out = json.loads(capsys.readouterr().out)
    assert [r["unit"] for r in out["toy"]["tandem"]] == ["MA", "A", "ST", "SG"]


def test_cli_text(capsys):
    assert main(["MAMAMAMAMAMAPRTEINQQQQQQQQQQ"]) == 0
    assert "tandem" in capsys.readouterr().out
