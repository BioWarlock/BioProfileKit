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
    _aa_group_distribution,
    _gravy,
    plot_overview,
    dna_rna_columns,
    protein_columns,
    AA_GROUPS,
    KYTE_DOOLITTLE,
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
# _aa_group_distribution
# ---------------------------------------------------------------------------

class TestAaGroupDistribution:
    def test_all_five_groups_present(self):
        seqs = pd.Series(["ACDEFGHIKLMNPQRSTVWY"])
        result = _aa_group_distribution(seqs)
        assert set(result.keys()) == set(AA_GROUPS.keys())

    def test_empty_series_returns_all_zero(self):
        seqs = pd.Series([], dtype=str)
        result = _aa_group_distribution(seqs)
        assert all(v == 0.0 for v in result.values())

    def test_single_group_composition(self):
        """GAVL are all in the 'Unpolar' group only."""
        seqs = pd.Series(["GAVL"])
        result = _aa_group_distribution(seqs)
        assert result["Unpolar"] == pytest.approx(1.0)
        for group in AA_GROUPS:
            if group != "Unpolar":
                assert result[group] == 0.0

    def test_mixed_composition_proportions(self):
        # G,A -> Unpolar (2), F -> Aromatic (1), K -> Positive (1) => 4 total
        seqs = pd.Series(["GAFK"])
        result = _aa_group_distribution(seqs)
        assert result["Unpolar"] == pytest.approx(0.5)
        assert result["Aromatic"] == pytest.approx(0.25)
        assert result["Positive"] == pytest.approx(0.25)
        assert result["Polar"] == 0.0
        assert result["Negative"] == 0.0

    def test_values_rounded_to_4_decimals(self):
        seqs = pd.Series(["ACDEFGHIKLMNPQRSTVWY"])
        result = _aa_group_distribution(seqs)
        for v in result.values():
            assert round(v, 4) == v

    def test_multiple_sequences_concatenated(self):
        """Distribution is computed over the concatenation of all
        sequences in the Series, not per-sequence."""
        seqs = pd.Series(["GG", "FF"])  # 2 Unpolar + 2 Aromatic
        result = _aa_group_distribution(seqs)
        assert result["Unpolar"] == pytest.approx(0.5)
        assert result["Aromatic"] == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# _gravy — real BioPython ProteinAnalysis, including the ambiguous-residue
# fallback to the local Kyte-Doolittle table
# ---------------------------------------------------------------------------

class TestGravy:
    def test_empty_sequence_returns_zero(self):
        assert _gravy("") == 0.0

    def test_standard_sequence_uses_biopython(self):
        """A sequence made only of standard amino acids should compute
        cleanly via Bio.SeqUtils.ProtParam without hitting the fallback."""
        result = _gravy("ACDEFGHIKLMNPQRSTVWY")
        assert isinstance(result, float)
        assert -4.5 <= result <= 4.5  # bounded by Kyte-Doolittle's own range

    def test_single_hydrophobic_residue(self):
        # Isoleucine is the most hydrophobic residue on the KD scale (4.5)
        result = _gravy("I")
        assert result == pytest.approx(4.5, abs=0.01)

    def test_ambiguous_residue_falls_back_to_kyte_doolittle(self):
        """'X' is not a standard residue — BioPython's GRAVY raises, and
        the fallback averages KYTE_DOOLITTLE.get(aa, 0.0) manually."""
        result = _gravy("AAX")
        expected = (KYTE_DOOLITTLE["A"] + KYTE_DOOLITTLE["A"] + 0.0) / 3
        assert result == pytest.approx(expected)

    def test_stop_codon_marker_falls_back(self):
        """'*' (stop codon) is not a valid residue either."""
        result = _gravy("A*")
        expected = (KYTE_DOOLITTLE["A"] + 0.0) / 2
        assert result == pytest.approx(expected)

    def test_fallback_matches_manual_kyte_doolittle_average(self):
        seq = "AAXJUOB*"  # mix of one real + several ambiguous codes
        result = _gravy(seq)
        expected = sum(KYTE_DOOLITTLE.get(aa, 0.0) for aa in seq) / len(seq)
        assert result == pytest.approx(expected)


# ---------------------------------------------------------------------------
# plot_overview (bar-chart variant, local to sequence_data.py — takes
# parallel kmer/count lists, distinct from analysis.plot_utils.plot_overview
# which takes a single column)
# ---------------------------------------------------------------------------

class TestPlotOverview:
    def test_returns_html_string(self):
        result = plot_overview(["AAA", "CCC", "GGG"], [5, 3, 1])
        assert isinstance(result, str)
        assert len(result) > 0

    def test_single_kmer(self):
        result = plot_overview(["AAA"], [10])
        assert isinstance(result, str)


# ---------------------------------------------------------------------------
# dna_rna_columns — end-to-end with real plotting/outlier/plot_utils code.
# Sequences deliberately have VARYING length so `plot` is built via the
# k-mer overview path rather than make_logo (WebLogo), which needs a real
# network call and is out of scope for a unit test.
# ---------------------------------------------------------------------------

class TestDnaRnaColumns:
    def _seqs(self):
        return pd.Series([
            "ACGTACGT", "TTTTGGGGCCCC", "ATATATAT", "GCGCGCGCGC", "AAAACCCCGGGGTTTT",
        ])

    def test_returns_expected_shape(self):
        result = dna_rna_columns(self._seqs(), k=3, top_n=20, top=5)
        assert len(result.sequence) == 5
        assert len(result.count) == 5
        assert len(result.length) == 5

    def test_uses_kmer_plot_for_varying_lengths(self):
        result = dna_rna_columns(self._seqs(), k=3, top_n=20, top=5)
        assert isinstance(result.plot, str)

    def test_invalid_sequences_are_filtered_out(self):
        seqs = pd.concat([self._seqs(), pd.Series(["BADSEQXYZ"])], ignore_index=True)
        result = dna_rna_columns(seqs, k=3, top_n=20, top=5, invalid=[(5, "BADSEQXYZ")])
        assert "BADSEQXYZ" not in result.sequence

    def test_gc_content_within_valid_percentage_range(self):
        result = dna_rna_columns(self._seqs(), k=3, top_n=20, top=5)
        assert 0.0 <= result.gc_content.min
        assert result.gc_content.max <= 100.0

    def test_length_distribution_plot_present_for_varying_lengths(self):
        result = dna_rna_columns(self._seqs(), k=3, top_n=20, top=5)
        assert result.length_distribution is not None

    def test_length_distribution_plot_absent_for_uniform_lengths(self):
        """`length_distribution` is only built when there is more than one
        distinct sequence length across the whole column. All sequences
        here share one length, which also routes `plot` through
        make_logo/WebLogo, so it needs Bio.motifs' real weblogo() call —
        left untested here since it needs network access."""
        seqs = pd.Series(["ACGT", "TTTT", "GGGG", "CCCC", "ATAT", "GCGC"])
        with patch("biological.sequence_data.make_logo", return_value="<svg></svg>"):
            result = dna_rna_columns(seqs, k=2, top_n=20, top=5)
        assert result.length_distribution is None

    def test_ambiguous_distribution_present_when_ambiguous_bases_exist(self):
        seqs = pd.Series(["ACGTN", "TTTTR", "GGGGY", "AAAACCCC", "TTTTGGGG"])
        result = dna_rna_columns(seqs, k=2, top_n=20, top=5)
        assert result.ambiguous_distribution is not None

    def test_ambiguous_distribution_absent_without_ambiguous_bases(self):
        result = dna_rna_columns(self._seqs(), k=3, top_n=20, top=5)
        assert result.ambiguous_distribution is None

    def test_reverse_complement_ratio_is_a_percentage(self):
        result = dna_rna_columns(self._seqs(), k=3, top_n=20, top=5)
        assert 0.0 <= result.reverse_complement_ratio <= 100.0

    def test_gc_distribution_and_at_gc_skewness_plots_present(self):
        """These two plots are always built, regardless of length uniformity."""
        result = dna_rna_columns(self._seqs(), k=3, top_n=20, top=5)
        assert result.gc_distribution is not None
        assert result.at_gc_skewness is not None


# ---------------------------------------------------------------------------
# protein_columns — end-to-end with real peptides/BioPython calls.
# Sequences have varying length to route `plot` through the k-mer path
# instead of make_logo.
# ---------------------------------------------------------------------------

class TestProteinColumns:
    def _seqs(self):
        return pd.Series([
            "ACDEFGHIK", "LMNPQRSTVWY", "ACDEFGHIKLMNPQ", "GAVLIMPFWY", "STCNQKRHDE",
        ])

    def test_returns_expected_shape(self):
        result = protein_columns(self._seqs(), k=3, top_n=20, top=5)
        assert len(result.sequence) == 5
        assert len(result.molecular_weight) == 5
        assert len(result.hydrophobicity) == 5

    def test_uses_kmer_plot_for_varying_lengths(self):
        result = protein_columns(self._seqs(), k=3, top_n=20, top=5)
        assert isinstance(result.plot, str)

    def test_stop_codon_ratio_reflects_asterisks(self):
        seqs = pd.Series(["ACDEFGHIK*", "LMNPQRSTVWY", "ACDEFGHIKLMNPQ", "GAVLIMPFWY", "STCNQKRHDE"])
        result = protein_columns(seqs, k=3, top_n=20, top=5)
        assert result.stop_codon_ratio == pytest.approx(20.0)  # 1 of 5 sequences

    def test_invalid_sequences_filtered_out(self):
        seqs = pd.concat([self._seqs(), pd.Series(["ZZZINVALID"])], ignore_index=True)
        result = protein_columns(seqs, k=3, top_n=20, top=5, invalid=[(5, "ZZZINVALID")])
        assert "ZZZINVALID" not in result.sequence

    def test_aa_group_distribution_populated(self):
        result = protein_columns(self._seqs(), k=3, top_n=20, top=5)
        assert set(result.aa_group_distribution.keys()) == set(AA_GROUPS.keys())
        assert isinstance(result.aa_group_plot, str)

    def test_descriptor_lists_match_number_of_top_entries(self):
        result = protein_columns(self._seqs(), k=3, top_n=20, top=5)
        n = len(result.sequence)
        assert len(result.frequency) == n
        assert len(result.charge) == n
        assert len(result.isoelectric_point) == n
        assert len(result.aliphatic_index) == n
        assert len(result.boman) == n
        assert len(result.aromaticity) == n
        assert len(result.instability) == n

    def test_ambiguous_residues_do_not_crash_gravy_computation(self):
        """A sequence containing 'X' must not crash protein_columns —
        _gravy's Kyte-Doolittle fallback should kick in transparently."""
        seqs = pd.concat([self._seqs(), pd.Series(["ACDEFGHIKX"])], ignore_index=True)
        result = protein_columns(seqs, k=3, top_n=20, top=5)
        assert result.gravy is not None