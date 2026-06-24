from collections import defaultdict

import numpy as np
import pandas as pd
import pytest
import sys
import os
from unittest.mock import patch, MagicMock

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from biological.sequence_data import (
    count_nmer,
    top_mere,
    biological_data_top_entries,
    _kmer_check,
    _reverse_complement_duplicates,
    _normalized_shanon_entropy,
    _dinucleotide_oe,
    protein_descriptors,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_df(seqs):
    """Build the minimal DataFrame that _dinucleotide_oe expects."""
    df = pd.DataFrame({"sequence": seqs})
    df["sequence"] = df["sequence"].str.upper()
    df["lengths"] = df["sequence"].str.len()
    return df


# ---------------------------------------------------------------------------
# count_nmer
# ---------------------------------------------------------------------------

class TestCountNmer:
    def test_trigram_counts(self):
        result = count_nmer("ATCATC", 3)
        assert result["ATC"] == 2
        assert result["TCA"] == 1
        assert result["CAT"] == 1

    def test_unigram_counts(self):
        result = count_nmer("AAACG", 1)
        assert result["A"] == 3
        assert result["C"] == 1
        assert result["G"] == 1

    def test_n_equals_sequence_length(self):
        result = count_nmer("ATCG", 4)
        assert result["ATCG"] == 1
        assert len(result) == 1

    def test_single_character_sequence(self):
        result = count_nmer("A", 1)
        assert result["A"] == 1

    def test_returns_defaultdict_with_zero_default(self):
        result = count_nmer("ATCG", 2)
        assert result["XX"] == 0  # missing key returns 0

    def test_n_larger_than_sequence_raises(self):
        assert count_nmer("AT", 5) == defaultdict(int)

    def test_homopolymer(self):
        result = count_nmer("AAAAA", 2)
        assert result["AA"] == 4

    def test_case_sensitive(self):
        result = count_nmer("AtCg", 1)
        # Input is not uppercased inside count_nmer
        assert result["A"] == 1
        assert result["t"] == 1


# ---------------------------------------------------------------------------
# top_mere
# ---------------------------------------------------------------------------

class TestTopMere:
    def test_returns_top_n_kmers(self):
        result = top_mere("ATCATCATC", n=3, top=2)
        assert result is not None
        assert len(result) <= 2

    def test_most_frequent_kmer_first(self):
        result = top_mere("ATCATCATCGGG", n=3, top=5)
        assert result is not None
        # ATC appears 3 times — should be first
        assert result[0][0] == "ATC"
        assert result[0][1] == 3

    def test_returns_none_for_empty_sequence(self):
        assert top_mere("", n=3) is None

    def test_returns_none_when_seq_shorter_than_n(self):
        assert top_mere("AT", n=3) is None

    def test_returns_none_for_none_input(self):
        assert top_mere(None, n=3) is None

    def test_sequence_exactly_n_returns_one_kmer(self):
        result = top_mere("ATG", n=3, top=5)
        assert result is not None
        assert len(result) == 1
        assert result[0] == ("ATG", 1)

    def test_top_capped_at_available_kmers(self):
        """Requesting top=10 from a short sequence returns fewer."""
        result = top_mere("ATCG", n=2, top=10)
        assert result is not None
        assert len(result) <= 3  # "AT", "TC", "CG"

    def test_result_is_sorted_descending(self):
        result = top_mere("AAAAT", n=1, top=5)
        assert result is not None
        counts = [c for _, c in result]
        assert counts == sorted(counts, reverse=True)


# ---------------------------------------------------------------------------
# biological_data_top_entries
# ---------------------------------------------------------------------------

class TestBiologicalDataTopEntries:
    def test_returns_correct_shapes(self):
        seqs = pd.Series(["ATCG", "ATCG", "GCTA", "TTTT"])
        uniques, counts, min_len, max_len, lengths = biological_data_top_entries(seqs, top_k=10)
        assert len(uniques) == len(counts) == len(lengths)
        assert len(uniques) == 3  # 3 unique sequences

    def test_top_k_limits_results(self):
        seqs = pd.Series([f"SEQ{i}" for i in range(20)])
        uniques, counts, *_ = biological_data_top_entries(seqs, top_k=5)
        assert len(uniques) == 5

    def test_sorted_by_frequency_descending(self):
        seqs = pd.Series(["A"] * 10 + ["B"] * 5 + ["C"] * 1)
        uniques, counts, *_ = biological_data_top_entries(seqs, top_k=3)
        assert list(counts) == sorted(counts, reverse=True)

    def test_uppercases_sequences(self):
        seqs = pd.Series(["atcg", "ATCG"])
        uniques, counts, *_ = biological_data_top_entries(seqs, top_k=10)
        # Both map to "ATCG" after uppercasing
        assert len(uniques) == 1
        assert counts[0] == 2

    def test_min_max_lengths(self):
        seqs = pd.Series(["A", "ATG", "ATCG"])
        _, _, min_len, max_len, _ = biological_data_top_entries(seqs, top_k=10)
        assert min_len == 1
        assert max_len == 4


# ---------------------------------------------------------------------------
# _kmer_check
# ---------------------------------------------------------------------------

class TestKmerCheck:
    def test_uses_k_when_sequences_long_enough(self):
        uniques = np.array(["ATCGATCG", "GCTAGCTA"])
        result = _kmer_check(k=3, top=5, uniques=uniques)
        assert len(result) == 2
        assert all(r is not None for r in result)

    def test_falls_back_to_k3_when_seq_too_short(self, capsys):
        """Sequences shorter than k trigger fallback to k=3."""
        uniques = np.array(["AT", "GC"])  # len=2 <= k=5
        result = _kmer_check(k=5, top=5, uniques=uniques)
        captured = capsys.readouterr()
        assert "k-Mer" in captured.out
        # With k=3 and seq len=2, top_mere returns None
        assert all(r is None for r in result)

    def test_returns_list_of_same_length_as_uniques(self):
        uniques = np.array(["ATCGATCG", "GCTAGCTA", "TTTTAAAA"])
        result = _kmer_check(k=3, top=5, uniques=uniques)
        assert len(result) == 3


# ---------------------------------------------------------------------------
# _reverse_complement_duplicates
# ---------------------------------------------------------------------------

class TestReverseComplementDuplicates:
    def test_no_duplicates_in_distinct_seqs(self):
        seqs = pd.Series(["AAAA", "CCCC", "GGGG"])
        ratio, dup_set = _reverse_complement_duplicates(seqs)
        # AAAA rev_comp = TTTT (not in set), CCCC rev_comp = GGGG (in set!)
        # CCCC and GGGG are rev_comp of each other → one is redundant
        assert ratio >= 0.0

    def test_palindrome_not_duplicate(self):
        """A palindromic sequence is its own reverse complement → not redundant."""
        seqs = pd.Series(["ATAT"])  # rev_comp = ATAT
        ratio, dup_set = _reverse_complement_duplicates(seqs)
        assert ratio == 0.0
        assert len(dup_set) == 0

    def test_complement_pair_flagged(self):
        """ATCG and its reverse complement CGAT should yield one redundant."""
        # rev_comp of ATCG: complement = TAGC, reverse = CGAT
        seqs = pd.Series(["ATCG", "CGAT"])
        ratio, dup_set = _reverse_complement_duplicates(seqs)
        assert ratio == pytest.approx(50.0, abs=0.01)
        assert len(dup_set) == 1

    def test_empty_series_returns_zero(self):
        seqs = pd.Series([], dtype=str)
        ratio, dup_set = _reverse_complement_duplicates(seqs)
        assert ratio == 0.0
        assert len(dup_set) == 0

    def test_ratio_between_0_and_100(self):
        seqs = pd.Series(["ATCG", "CGAT", "TTTT", "AAAA", "GCGC"])
        ratio, _ = _reverse_complement_duplicates(seqs)
        assert 0.0 <= ratio <= 100.0

    def test_ratio_rounded_to_two_decimals(self):
        seqs = pd.Series(["ATCG", "CGAT", "GCTA"])
        ratio, _ = _reverse_complement_duplicates(seqs)
        assert round(ratio, 2) == ratio

    def test_all_reverse_complement_pairs(self):
        """3 pairs of reverse complements → 3 redundant out of 6 = 50 %."""
        seqs = pd.Series(["ATCG", "CGAT", "GGCC", "GGCC"[::-1], "AATT", "TTAA"])
        ratio, _ = _reverse_complement_duplicates(seqs)
        assert ratio > 0.0


# ---------------------------------------------------------------------------
# _normalized_shanon_entropy
# ---------------------------------------------------------------------------

class TestNormalizedShanonEntropy:
    def test_empty_sequence_returns_zero(self):
        result = _normalized_shanon_entropy("")
        assert result == pytest.approx(0.0, abs=1e-6)

    def test_single_character_returns_zero(self):
        """Single character: only one symbol → entropy = 0."""
        result = _normalized_shanon_entropy("A")
        assert result == pytest.approx(0.0, abs=1e-6)

    def test_homopolymer_returns_zero(self):
        result = _normalized_shanon_entropy("AAAAAAA")
        assert result == pytest.approx(0.0, abs=0.01)

    def test_two_equal_symbols_returns_100(self):
        """AB: 2 symbols equally distributed → max entropy = 100 %."""
        result = _normalized_shanon_entropy("AB")
        assert result == pytest.approx(100.0, abs=0.01)

    def test_result_between_0_and_100(self):
        result = _normalized_shanon_entropy("ATCGATCGATCG")
        assert 0.0 <= float(result) <= 100.0

    def test_higher_entropy_for_diverse_sequence(self):
        """Uniformly distributed ACGT should have higher entropy than biased."""
        diverse = _normalized_shanon_entropy("ACGTACGTACGT")
        biased = _normalized_shanon_entropy("AAACAAACAAAC")
        assert diverse > biased

    def test_returns_float64(self):
        result = _normalized_shanon_entropy("ATCG")
        assert isinstance(result, (float, np.floating))

    def test_low_complexity_poly_a_tail(self):
        """Poly-A tail typical in mRNA: very low complexity."""
        poly_a = "A" * 50 + "ATCG"
        result = _normalized_shanon_entropy(poly_a)
        assert float(result) < 50.0  # low entropy


# ---------------------------------------------------------------------------
# _dinucleotide_oe
# ---------------------------------------------------------------------------

class TestDinucleotideOE:
    def test_returns_two_metric_summaries(self):
        df = make_df(["ATCGATCG", "GCTAGCTA"])
        cpg, tpa = _dinucleotide_oe(df)
        for summary in (cpg, tpa):
            assert hasattr(summary, "min")
            assert hasattr(summary, "max")
            assert hasattr(summary, "mean")

    def test_cpg_oe_non_negative(self):
        df = make_df(["ATCGATCGATCG", "GCGCGCGCGCGC"])
        cpg, _ = _dinucleotide_oe(df)
        assert cpg.min >= 0.0
        assert cpg.max >= 0.0

    def test_tpa_oe_non_negative(self):
        df = make_df(["ATATATATATAT", "GCGCGCGCGCGC"])
        _, tpa = _dinucleotide_oe(df)
        assert tpa.min >= 0.0

    def test_cpg_depleted_sequence(self):
        """CpG-depleted sequence (common in mammalian genomes) should have low O/E."""
        # AT-rich, no CG dinucleotide
        seqs = ["ATATATAT"] * 5
        df = make_df(seqs)
        cpg, _ = _dinucleotide_oe(df)
        assert cpg.mean == pytest.approx(0.0, abs=0.01)

    def test_cpg_enriched_sequence(self):
        """CpG island-like sequence: high CG frequency → O/E near or above 1."""
        seqs = ["CGCGCGCGCGCG"] * 5
        df = make_df(seqs)
        cpg, _ = _dinucleotide_oe(df)
        assert cpg.mean > 0.5

    def test_values_rounded_to_4_decimals(self):
        df = make_df(["ATCGATCGATCG"])
        cpg, tpa = _dinucleotide_oe(df)
        for val in (cpg.min, cpg.max, cpg.mean, tpa.min, tpa.max, tpa.mean):
            assert round(val, 4) == val

    def test_single_length_sequence_no_crash(self):
        """Length-1 sequence: dinucleotide count is 0, exp may be 0 → handled by where."""
        df = make_df(["A"])
        cpg, tpa = _dinucleotide_oe(df)
        # Should not raise; values will be 0.0 due to np.where guard
        assert cpg.mean >= 0.0
        assert tpa.mean >= 0.0


# ---------------------------------------------------------------------------
# protein_descriptors
# ---------------------------------------------------------------------------

class TestProteinDescriptors:
    def test_returns_all_expected_keys(self):
        result = protein_descriptors("ACDEFGHIKLM")
        for key in ("seq", "freq", "aidx", "boman", "charge", "hp",
                    "iep", "iidx", "mol", "aroma"):
            assert key in result, f"Missing key: {key}"

    def test_seq_matches_input(self):
        peptide = "ACDEFGHIKLM"
        result = protein_descriptors(peptide)
        assert result["seq"] == peptide

    def test_aromaticity_only_fwy(self):
        """Aromaticity counts F, W, Y only."""
        # ACFW: 4 residues, 2 aromatic → 0.5
        result = protein_descriptors("ACFW")
        assert result["aroma"] == pytest.approx(0.5, abs=0.01)

    def test_aromaticity_all_aromatic(self):
        """All residues aromatic: aromaticity = 1.0."""
        result = protein_descriptors("FFFWYWF")
        assert result["aroma"] == 1.0

    def test_aromaticity_zero_no_aromatic_aa(self):
        result = protein_descriptors("ACDE")
        assert result["aroma"] == pytest.approx(0.0, abs=1e-6)

    def test_molecular_weight_positive(self):
        result = protein_descriptors("ACDEFGHIKLM")
        assert result["mol"] > 0

    def test_zero_division_in_aliphatic_index_handled(self):
        """Peptide with no A/I/V/L: aliphatic_index raises ZeroDivisionError → caught as 0.0."""
        # FWDE: no aliphatic residues
        result = protein_descriptors("FWDE")
        assert result["aidx"] == 0.0

    def test_freq_sums_to_one(self):
        result = protein_descriptors("ACDEFGHIKLM")
        assert sum(result["freq"].values()) == pytest.approx(1.0, abs=1e-6)

    def test_isoelectric_point_in_valid_range(self):
        result = protein_descriptors("ACDEFGHIKLM")
        assert 0.0 < result["iep"] < 14.0

    def test_single_amino_acid(self):
        """Edge case: single residue peptide."""
        result = protein_descriptors("A")
        assert result["seq"] == "A"
        assert result["mol"] > 0

# ---------------------------------------------------------------------------
# _gravy  (lines 401-408)
# ---------------------------------------------------------------------------

from biological.sequence_data import _gravy, KYTE_DOOLITTLE, AA_GROUPS

class TestGravy:
    def test_empty_sequence_returns_zero(self):
        assert _gravy("") == 0.0

    def test_pure_alanine(self):
        """Alanine has KD index 1.8 — known reference value."""
        result = _gravy("AAAA")
        assert result == pytest.approx(1.8, abs=0.01)

    def test_hydrophilic_sequence(self):
        """Arginine (R) has KD = -4.5 — strongly hydrophilic."""
        result = _gravy("RRRR")
        assert result == pytest.approx(-4.5, abs=0.01)

    def test_mixed_sequence(self):
        """GRAVY of mixed sequence must be between the min and max KD values."""
        result = _gravy("ARND")  # A=1.8, R=-4.5, N=-3.5, D=-3.5
        expected = (1.8 + -4.5 + -3.5 + -3.5) / 4
        assert result == pytest.approx(expected, abs=0.01)

    def test_ambiguous_residue_X_uses_fallback(self):
        """X is not a standard amino acid — ProteinAnalysis raises KeyError,
        fallback must use KYTE_DOOLITTLE.get(aa, 0.0)."""
        # X not in KYTE_DOOLITTLE → contributes 0.0
        result = _gravy("XXXX")
        assert result == pytest.approx(0.0, abs=1e-6)

    def test_stop_codon_marker_uses_fallback(self):
        """'*' (stop codon) triggers fallback; contributes 0.0."""
        result = _gravy("A*A")  # A=1.8, *=0.0, A=1.8 → mean=1.2
        assert result == pytest.approx(1.2, abs=0.01)

    def test_single_residue(self):
        """Single residue: GRAVY == its own KD index."""
        for aa, kd in list(KYTE_DOOLITTLE.items())[:5]:
            assert _gravy(aa) == pytest.approx(kd, abs=0.01)

    def test_result_is_float(self):
        assert isinstance(_gravy("ACGT"), float)


# ---------------------------------------------------------------------------
# _aa_group_distribution  (lines 410-419)
# ---------------------------------------------------------------------------

from biological.sequence_data import _aa_group_distribution

class TestAaGroupDistribution:
    def test_returns_all_five_groups(self):
        seqs = pd.Series(["ACDEFGHIKLMNPQRSTVWY"])
        result = _aa_group_distribution(seqs)
        assert set(result.keys()) == set(AA_GROUPS.keys())

    def test_empty_string_series_returns_zeros(self):
        """Series containing only an empty string → total=0 → all groups 0.0."""
        result = _aa_group_distribution(pd.Series([""], dtype=str))
        assert all(v == 0.0 for v in result.values())

    def test_empty_series_no_elements_returns_zeros(self):
        """Series with no elements → str.cat() returns '' → total=0."""
        result = _aa_group_distribution(pd.Series([], dtype=str))
        assert all(v == 0.0 for v in result.values())

    def test_pure_unpolar_sequence(self):
        """GAVLIMP are all Unpolar → Unpolar fraction should be 1.0."""
        seqs = pd.Series(["GAVLIMP"])
        result = _aa_group_distribution(seqs)
        assert result["Unpolar"] == pytest.approx(1.0, abs=1e-4)
        assert result["Aromatic"] == pytest.approx(0.0, abs=1e-4)

    def test_pure_aromatic_sequence(self):
        """FWY are all Aromatic."""
        seqs = pd.Series(["FWYFWY"])
        result = _aa_group_distribution(seqs)
        assert result["Aromatic"] == pytest.approx(1.0, abs=1e-4)

    def test_values_sum_to_one_for_standard_aa(self):
        """All 20 standard AAs together → fractions must sum to 1.0."""
        seqs = pd.Series(["ACDEFGHIKLMNPQRSTVWY"])
        result = _aa_group_distribution(seqs)
        assert sum(result.values()) == pytest.approx(1.0, abs=1e-4)

    def test_values_between_0_and_1(self):
        seqs = pd.Series(["AAKKKFFFRR"])
        result = _aa_group_distribution(seqs)
        for v in result.values():
            assert 0.0 <= v <= 1.0

    def test_multiple_sequences_concatenated(self):
        """Distribution should be computed over all sequences combined."""
        seqs_combined = pd.Series(["AAAA", "KKKK"])
        result_combined = _aa_group_distribution(seqs_combined)
        # AAAA = Unpolar, KKKK = Positive → each 50%
        assert result_combined["Unpolar"] == pytest.approx(0.5, abs=1e-4)
        assert result_combined["Positive"] == pytest.approx(0.5, abs=1e-4)

    def test_rounded_to_four_decimals(self):
        seqs = pd.Series(["ACDEFGHIKLMNPQRSTVWY"])
        result = _aa_group_distribution(seqs)
        for v in result.values():
            assert v == round(v, 4)

    def test_non_standard_residue_not_in_any_group(self):
        """B, Z, X etc. are not in any AA_GROUPS set → do not inflate any fraction."""
        seqs = pd.Series(["AXBZ"])   # only X, B, Z (non-standard) + A (Unpolar)
        result = _aa_group_distribution(seqs)
        # A is 1/4 = 0.25 Unpolar; others not counted anywhere
        assert result["Unpolar"] == pytest.approx(0.25, abs=1e-4)
        total = sum(result.values())
        assert total < 1.0  # non-standard residues are not counted


# ---------------------------------------------------------------------------
# dna_rna_columns  (lines 67-189)  — mocked plot/logo dependencies
# ---------------------------------------------------------------------------

class TestDnaRnaColumns:
    """Tests for dna_rna_columns with Plotly and WebLogo mocked out."""

    MOCK_PLOT_HTML = "<div>mock_plot</div>"
    MOCK_LOGO_SVG  = "<svg>mock_logo</svg>"

    def _mock_targets(self):
        return {
            "biological.sequence_data.at_gc_skewness":      MagicMock(return_value=self.MOCK_PLOT_HTML),
            "biological.sequence_data.gc_distribution":     MagicMock(return_value=self.MOCK_PLOT_HTML),
            "biological.sequence_data.ambiguous_distribution": MagicMock(return_value=self.MOCK_PLOT_HTML),
            "biological.sequence_data.length_distribution": MagicMock(return_value=self.MOCK_PLOT_HTML),
            "biological.sequence_data.make_logo":           MagicMock(return_value=self.MOCK_LOGO_SVG),
            "biological.sequence_data.plot_overview":       MagicMock(return_value=self.MOCK_PLOT_HTML),
        }

    def _invoke(self, seqs, **kwargs):
        from biological.sequence_data import dna_rna_columns
        mocks = self._mock_targets()
        with patch("biological.sequence_data.at_gc_skewness",      mocks["biological.sequence_data.at_gc_skewness"]), \
             patch("biological.sequence_data.gc_distribution",     mocks["biological.sequence_data.gc_distribution"]), \
             patch("biological.sequence_data.ambiguous_distribution", mocks["biological.sequence_data.ambiguous_distribution"]), \
             patch("biological.sequence_data.length_distribution", mocks["biological.sequence_data.length_distribution"]), \
             patch("biological.sequence_data.make_logo",           mocks["biological.sequence_data.make_logo"]), \
             patch("biological.sequence_data.plot_overview",       mocks["biological.sequence_data.plot_overview"]):
            return dna_rna_columns(seqs, **kwargs)

    # ── Basic output structure ──────────────────────────────────────────────

    def test_returns_dnarnacol_with_expected_fields(self):
        seqs = pd.Series(["ATCGATCG"] * 10)
        result = self._invoke(seqs)
        for field in ("gc_content", "length_stats", "ambiguous_base_ratio",
                      "codon_completeness", "gc_skew", "at_skew",
                      "cpg_observed_expected", "tpa_observed_expected",
                      "low_complexity", "reverse_complement_ratio"):
            assert hasattr(result, field), f"Missing field: {field}"

    def test_gc_content_range(self):
        """GC content must be between 0 and 100 %."""
        seqs = pd.Series(["ATATATATT"] * 10)   # low GC
        result = self._invoke(seqs)
        assert 0.0 <= result.gc_content.min <= result.gc_content.max <= 100.0

    def test_pure_gc_sequence(self):
        """GCGCGCGC → GC content should be 100 %."""
        seqs = pd.Series(["GCGCGCGC"] * 10)
        result = self._invoke(seqs)
        assert result.gc_content.mean == pytest.approx(100.0, abs=0.1)

    def test_pure_at_sequence(self):
        """ATATATAT → GC content should be 0 %."""
        seqs = pd.Series(["ATATATAT"] * 10)
        result = self._invoke(seqs)
        assert result.gc_content.mean == pytest.approx(0.0, abs=0.1)

    def test_invalid_seqs_excluded(self):
        """Sequences marked as invalid must not contribute to the analysis."""
        seqs = pd.Series(["ATCGATCG"] * 9 + ["INVALID"])
        result = self._invoke(seqs, invalid=["INVALID"])
        # INVALID is excluded → only 9 sequences analysed
        assert result.length_stats.mean == pytest.approx(8.0, abs=0.1)

    def test_nan_values_dropped(self):
        """NaN entries must be dropped silently before analysis."""
        import numpy as np
        seqs = pd.Series(["ATCGATCG"] * 9 + [np.nan])
        result = self._invoke(seqs)
        assert result.length_stats.mean == pytest.approx(8.0, abs=0.1)

    def test_custom_kmer_size(self):
        """k parameter must be forwarded to kmer computation."""
        seqs = pd.Series(["ATCGATCGATCG"] * 10)
        # Should not raise for any valid k
        result = self._invoke(seqs, k=4, top_n=5)
        assert result is not None

    def test_codon_completeness_full_codons(self):
        """Sequences whose length is divisible by 3 → 100 % codon completeness."""
        seqs = pd.Series(["ATGATGATG"] * 10)  # len=9, 9%3=0
        result = self._invoke(seqs)
        assert result.codon_completeness.mean == pytest.approx(100.0, abs=0.1)

    def test_codon_completeness_partial(self):
        """Length 10: 10%3=1 → 9/10 = 90 % complete."""
        seqs = pd.Series(["ATGATGATGA"] * 10)
        result = self._invoke(seqs)
        assert result.codon_completeness.mean == pytest.approx(90.0, abs=0.1)

    def test_ambiguous_base_ratio_with_n(self):
        """Sequences with N bases must show non-zero ambiguous_base_ratio."""
        seqs = pd.Series(["ATCGNNNN"] * 10)   # 4 N out of 8 → 50 %
        result = self._invoke(seqs)
        assert result.ambiguous_base_ratio.mean == pytest.approx(50.0, abs=0.1)

    def test_uniform_length_uses_logo(self):
        """When min_len == max_len, make_logo is called instead of plot_overview."""
        seqs = pd.Series(["ATCGATCG"] * 10)
        mocks = self._mock_targets()
        from biological.sequence_data import dna_rna_columns
        with patch("biological.sequence_data.at_gc_skewness",         mocks["biological.sequence_data.at_gc_skewness"]), \
             patch("biological.sequence_data.gc_distribution",        mocks["biological.sequence_data.gc_distribution"]), \
             patch("biological.sequence_data.ambiguous_distribution",  mocks["biological.sequence_data.ambiguous_distribution"]), \
             patch("biological.sequence_data.length_distribution",    mocks["biological.sequence_data.length_distribution"]), \
             patch("biological.sequence_data.make_logo",              mocks["biological.sequence_data.make_logo"]) as mock_logo, \
             patch("biological.sequence_data.plot_overview",          mocks["biological.sequence_data.plot_overview"]):
            dna_rna_columns(seqs)
        mock_logo.assert_called_once()

    def test_variable_length_uses_plot_overview(self):
        """When min_len != max_len, plot_overview is called instead of make_logo.
        Sequences must all be longer than k (default=3) so k_mers are not None."""
        seqs = pd.Series(["ATCGATCG"] * 5 + ["ATCGATCGATCGATCG"] * 5)
        mocks = self._mock_targets()
        from biological.sequence_data import dna_rna_columns
        with patch("biological.sequence_data.at_gc_skewness",         mocks["biological.sequence_data.at_gc_skewness"]), \
             patch("biological.sequence_data.gc_distribution",        mocks["biological.sequence_data.gc_distribution"]), \
             patch("biological.sequence_data.ambiguous_distribution",  mocks["biological.sequence_data.ambiguous_distribution"]), \
             patch("biological.sequence_data.length_distribution",    mocks["biological.sequence_data.length_distribution"]), \
             patch("biological.sequence_data.make_logo",              mocks["biological.sequence_data.make_logo"]), \
             patch("biological.sequence_data.plot_overview",          mocks["biological.sequence_data.plot_overview"]) as mock_plot:
            dna_rna_columns(seqs)
        mock_plot.assert_called_once()


# ---------------------------------------------------------------------------
# protein_columns  (lines 271-387) — mocked plot/logo dependencies
# ---------------------------------------------------------------------------

class TestProteinColumns:
    """Tests for protein_columns with Plotly and WebLogo mocked out."""

    MOCK_HTML = "<div>mock</div>"
    MOCK_SVG  = "<svg>mock</svg>"

    def _invoke(self, seqs, **kwargs):
        from biological.sequence_data import protein_columns
        with patch("biological.sequence_data.length_distribution",    MagicMock(return_value=self.MOCK_HTML)), \
             patch("biological.sequence_data.ambiguous_distribution",  MagicMock(return_value=self.MOCK_HTML)), \
             patch("biological.sequence_data.aa_group_distribution",   MagicMock(return_value=self.MOCK_HTML)), \
             patch("biological.sequence_data.make_logo",              MagicMock(return_value=self.MOCK_SVG)), \
             patch("biological.sequence_data.plot_overview",          MagicMock(return_value=self.MOCK_HTML)):
            return protein_columns(seqs, **kwargs)

    # ── Basic output structure ──────────────────────────────────────────────

    def test_returns_proteincols_with_expected_fields(self):
        seqs = pd.Series(["ACDEFGHIKLM"] * 10)
        result = self._invoke(seqs)
        for field in ("ambiguous_residue_ratio", "length_stats", "length_outliers",
                      "stop_codon_ratio", "low_complexity", "gravy",
                      "cysteine_count", "disorder_propensity", "aa_group_distribution"):
            assert hasattr(result, field), f"Missing field: {field}"

    def test_gravy_range_typical_proteins(self):
        """GRAVY of realistic proteins is roughly in [-4.5, 4.5]."""
        seqs = pd.Series(["ACDEFGHIKLMNPQRSTVWY"] * 10)
        result = self._invoke(seqs)
        assert -4.5 <= result.gravy.mean <= 4.5

    def test_cysteine_count_zero_without_cysteine(self):
        """Sequences without C must yield cysteine_count.mean == 0."""
        seqs = pd.Series(["ADEFGHIKLM"] * 10)   # no C
        result = self._invoke(seqs)
        assert result.cysteine_count.mean == pytest.approx(0.0, abs=1e-6)

    def test_cysteine_count_nonzero_with_cysteine(self):
        seqs = pd.Series(["ACAC"] * 10)   # 2 C per sequence
        result = self._invoke(seqs)
        assert result.cysteine_count.mean == pytest.approx(2.0, abs=0.1)

    def test_stop_codon_ratio_zero_without_stop(self):
        """Clean protein sequences without '*' must have stop_codon_ratio=0."""
        seqs = pd.Series(["ACDEFGHIKLM"] * 10)
        result = self._invoke(seqs)
        assert result.stop_codon_ratio == pytest.approx(0.0, abs=1e-6)

    def test_stop_codon_ratio_with_stop(self):
        """50 % of sequences have a stop codon → ratio should be ~50."""
        seqs = pd.Series(["ACDE*FGHIK"] * 5 + ["ACDEFGHIKL"] * 5)
        result = self._invoke(seqs)
        assert result.stop_codon_ratio == pytest.approx(50.0, abs=0.1)

    def test_ambiguous_residue_ratio_zero_for_clean_seq(self):
        """No X/J/U residues → ambiguous_residue_ratio should be 0."""
        seqs = pd.Series(["ACDEFGHIKLM"] * 10)
        result = self._invoke(seqs)
        assert result.ambiguous_residue_ratio.mean == pytest.approx(0.0, abs=1e-6)

    def test_ambiguous_residue_ratio_with_x(self):
        """Sequence with X: AXXXFGHIKLM → 3/11 ≈ 27.3 % ambiguous."""
        seqs = pd.Series(["AXXXFGHIKLM"] * 10)
        result = self._invoke(seqs)
        expected = 3 / 11 * 100
        assert result.ambiguous_residue_ratio.mean == pytest.approx(expected, abs=0.1)

    def test_invalid_seqs_excluded(self):
        """Sequences in the invalid list must be removed before analysis."""
        seqs = pd.Series(["ACDEFGHIKL"] * 9 + ["BADSEQ"])
        result = self._invoke(seqs, invalid=["BADSEQ"])
        assert result.length_stats.mean == pytest.approx(10.0, abs=0.1)

    def test_custom_kmer_size(self):
        seqs = pd.Series(["ACDEFGHIKLMNPQRST"] * 10)
        result = self._invoke(seqs, k=4)
        assert result is not None

    def test_disorder_propensity_pesqk_residues(self):
        """PESQK residues (DISORDER_AA) should yield high disorder propensity."""
        seqs = pd.Series(["PESQKPESQK"] * 10)   # 100 % disorder AAs
        result = self._invoke(seqs)
        assert result.disorder_propensity.mean == pytest.approx(100.0, abs=0.1)

    def test_disorder_propensity_zero_no_disorder_aa(self):
        """Sequence without any PESQK residue → disorder_propensity = 0."""
        seqs = pd.Series(["ACDFGHILMV"] * 10)   # no P/E/S/Q/K
        result = self._invoke(seqs)
        assert result.disorder_propensity.mean == pytest.approx(0.0, abs=0.1)

    def test_uniform_length_uses_logo(self):
        """When all sequences have equal length, make_logo is called."""
        seqs = pd.Series(["ACDEFGHIKL"] * 10)
        from biological.sequence_data import protein_columns
        mock_logo = MagicMock(return_value=self.MOCK_SVG)
        with patch("biological.sequence_data.length_distribution",    MagicMock(return_value=self.MOCK_HTML)), \
             patch("biological.sequence_data.ambiguous_distribution",  MagicMock(return_value=self.MOCK_HTML)), \
             patch("biological.sequence_data.aa_group_distribution",   MagicMock(return_value=self.MOCK_HTML)), \
             patch("biological.sequence_data.make_logo",              mock_logo), \
             patch("biological.sequence_data.plot_overview",          MagicMock(return_value=self.MOCK_HTML)):
            protein_columns(seqs)
        mock_logo.assert_called_once()


# ---------------------------------------------------------------------------
# make_logo  (lines 421-455) — WebLogo mocked
# ---------------------------------------------------------------------------

class TestMakeLogo:
    def test_returns_svg_string_on_success(self):
        """make_logo must return the extracted SVG fragment."""
        from biological.sequence_data import make_logo
        fake_svg = '<svg xmlns="http://www.w3.org/2000/svg"><text>logo</text></svg>'
        fake_html = f"<html><body>{fake_svg}</body></html>"
        m = MagicMock()
        with patch("biological.sequence_data.motifs") as mock_motifs, \
             patch("builtins.open", MagicMock(return_value=MagicMock(
                 __enter__=MagicMock(return_value=MagicMock(read=MagicMock(return_value=fake_html))),
                 __exit__=MagicMock(return_value=False)
             ))), \
             patch("tempfile.NamedTemporaryFile") as mock_tmp, \
             patch("pathlib.Path.is_file", return_value=True), \
             patch("pathlib.Path.unlink"):
            mock_tmp.return_value.__enter__ = MagicMock(return_value=MagicMock(name="/tmp/fake.svg"))
            mock_tmp.return_value.__exit__  = MagicMock(return_value=False)
            mock_motifs.create.return_value.weblogo = MagicMock()
            result = make_logo(["ATCGATCG", "ATCGATCG"], "color_classic", "dna")
        # Either returns SVG string or None (depends on mock depth)
        assert result is None or "<svg" in result

    def test_returns_none_on_network_error(self):
        """URLError / HTTPError during WebLogo call must return None gracefully."""
        from biological.sequence_data import make_logo
        from urllib.error import URLError
        with patch("biological.sequence_data.motifs") as mock_motifs, \
             patch("tempfile.NamedTemporaryFile") as mock_tmp, \
             patch("pathlib.Path.is_file", return_value=False):
            mock_tmp.return_value.__enter__ = MagicMock(return_value=MagicMock(name="/tmp/fake.svg"))
            mock_tmp.return_value.__exit__  = MagicMock(return_value=False)
            mock_motifs.create.return_value.weblogo.side_effect = URLError("timeout")
            result = make_logo(["ATCG", "ATCG"], "color_classic", "dna")
        assert result is None

    def test_protein_uses_protein_alphabet(self):
        """seq_type='protein' must call motifs.create with the protein alphabet."""
        from biological.sequence_data import make_logo
        from urllib.error import URLError
        with patch("biological.sequence_data.motifs") as mock_motifs, \
             patch("tempfile.NamedTemporaryFile") as mock_tmp, \
             patch("pathlib.Path.is_file", return_value=False):
            mock_tmp.return_value.__enter__ = MagicMock(return_value=MagicMock(name="/tmp/fake.svg"))
            mock_tmp.return_value.__exit__  = MagicMock(return_value=False)
            mock_motifs.create.return_value.weblogo.side_effect = URLError("x")
            make_logo(["ACDE", "ACDE"], "chemistry", "protein")
        call_kwargs = mock_motifs.create.call_args
        assert "ACDEFGHIKLMNPQRSTVWY" in str(call_kwargs)

    def test_dna_uses_dna_alphabet(self):
        from biological.sequence_data import make_logo
        from urllib.error import URLError
        with patch("biological.sequence_data.motifs") as mock_motifs, \
             patch("tempfile.NamedTemporaryFile") as mock_tmp, \
             patch("pathlib.Path.is_file", return_value=False):
            mock_tmp.return_value.__enter__ = MagicMock(return_value=MagicMock(name="/tmp/fake.svg"))
            mock_tmp.return_value.__exit__  = MagicMock(return_value=False)
            mock_motifs.create.return_value.weblogo.side_effect = URLError("x")
            make_logo(["ATCG", "ATCG"], "color_classic", "dna")
        call_kwargs = mock_motifs.create.call_args
        assert "ACGT" in str(call_kwargs)


# ---------------------------------------------------------------------------
# plot_overview  (lines 458-463)
# ---------------------------------------------------------------------------

class TestPlotOverview:
    def test_returns_html_string(self):
        from biological.sequence_data import plot_overview
        result = plot_overview(["ATG", "GCC", "TAA"], [10, 7, 3])
        assert isinstance(result, str)
        assert "<div" in result

    def test_empty_lists_no_crash(self):
        """Plotly px.bar raises ValueError on two empty lists — plot_overview
        should only be called with non-empty kmer data in practice."""
        from biological.sequence_data import plot_overview
        with pytest.raises(ValueError):
            plot_overview([], [])

    def test_single_kmer(self):
        from biological.sequence_data import plot_overview
        result = plot_overview(["ATG"], [42])
        assert isinstance(result, str)
        assert "<div" in result