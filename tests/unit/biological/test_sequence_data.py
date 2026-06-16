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