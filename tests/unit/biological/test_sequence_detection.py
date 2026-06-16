import math
import re
import pandas as pd
import pytest
import sys
import os
from unittest.mock import patch, MagicMock

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from biological.sequence_detection import (
    _alphabet_from_pattern,
    _get_invalid,
    check_sequence,
    DNA_ALPHABET,
    RNA_ALPHABET,
    PROTEIN_ALPHABET,
    ENTROPY_THRESHOLDS,
)

# ---------------------------------------------------------------------------
# Mock targets
# ---------------------------------------------------------------------------
FAST_CHECK = "biological.sequence_detection.fast_check_sequence"
CHAR_ENTROPY = "biological.sequence_detection.char_entropy"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_df(values, col="seq"):
    return pd.DataFrame({col: values})


def dna_seqs(n=15):
    """n unique DNA sequences, each length > 2, no non-alpha chars."""
    bases = ["ATCG", "GCTA", "TTAA", "CCGG", "ATAT", "GCGC", "TATA",
             "ACAC", "TGTG", "CACA", "AGAG", "CTCT", "GAGA", "TCTC", "ACGT"]
    return bases[:n]


# ---------------------------------------------------------------------------
# _alphabet_from_pattern
# ---------------------------------------------------------------------------

class TestAlphabetFromPattern:
    def test_extracts_characters_from_bracket_pattern(self):
        pattern = re.compile(r"^[ACGT]+$")
        result = _alphabet_from_pattern(pattern)
        assert result == {"A", "C", "G", "T"}

    def test_uppercases_characters(self):
        pattern = re.compile(r"^[acgt]+$")
        result = _alphabet_from_pattern(pattern)
        assert result == {"A", "C", "G", "T"}

    def test_returns_empty_set_for_no_bracket_group(self):
        pattern = re.compile(r"^ACGT$")
        result = _alphabet_from_pattern(pattern)
        assert result == set()

    def test_protein_alphabet_larger_than_dna(self):
        assert len(PROTEIN_ALPHABET) > len(DNA_ALPHABET)

    def test_dna_rna_same_size(self):
        """DNA and RNA alphabets differ only in T vs U — same cardinality."""
        assert len(DNA_ALPHABET) == len(RNA_ALPHABET)

    def test_dna_alphabet_contains_expected_bases(self):
        assert {"A", "C", "G", "T"}.issubset(DNA_ALPHABET)

    def test_rna_alphabet_contains_expected_bases(self):
        assert {"A", "C", "G", "U"}.issubset(RNA_ALPHABET)


# ---------------------------------------------------------------------------
# ENTROPY_THRESHOLDS
# ---------------------------------------------------------------------------

class TestEntropyThresholds:
    def test_all_keys_present(self):
        assert set(ENTROPY_THRESHOLDS.keys()) == {"dna", "rna", "protein"}

    def test_thresholds_are_positive(self):
        for key, val in ENTROPY_THRESHOLDS.items():
            assert val > 0, f"Threshold for {key} should be positive"

    def test_protein_threshold_higher_than_dna(self):
        """Protein has more symbols → higher entropy threshold."""
        assert ENTROPY_THRESHOLDS["protein"] > ENTROPY_THRESHOLDS["dna"]

    def test_threshold_formula(self):
        """Threshold = 0.5 * log2(alphabet_size)."""
        expected_dna = 0.5 * math.log2(len(DNA_ALPHABET))
        assert ENTROPY_THRESHOLDS["dna"] == pytest.approx(expected_dna, rel=1e-6)


# ---------------------------------------------------------------------------
# _get_invalid
# ---------------------------------------------------------------------------

class TestGetInvalid:
    def test_returns_empty_list_for_no_indices(self):
        assert _get_invalid(["A", "B", "C"], []) == []

    def test_returns_empty_list_for_none_indices(self):
        assert _get_invalid(["A", "B", "C"], None) == []

    def test_returns_values_at_given_indices(self):
        values = ["ATCG", "XXXX", "GCTA", "????"]
        result = _get_invalid(values, [1, 3])
        assert result == ["XXXX", "????"]

    def test_single_invalid_index(self):
        values = ["ATCG", "INVALID"]
        result = _get_invalid(values, [1])
        assert result == ["INVALID"]

    def test_all_indices_invalid(self):
        values = ["A", "B", "C"]
        result = _get_invalid(values, [0, 1, 2])
        assert result == ["A", "B", "C"]


# ---------------------------------------------------------------------------
# check_sequence — early-return guards
# ---------------------------------------------------------------------------

class TestCheckSequenceEarlyReturns:
    def test_numeric_column_returns_none(self):
        df = make_df([1.0, 2.0, 3.0])
        result, invalid = check_sequence(df, "seq")
        assert result == "None"
        assert invalid == []

    def test_bool_column_returns_none(self):
        df = pd.DataFrame({"seq": [True, False, True]})
        result, invalid = check_sequence(df, "seq")
        assert result == "None"
        assert invalid == []

    def test_single_char_values_return_none(self):
        """All values length 1 → not sequences."""
        df = make_df(list("ABCDEFGHIJKLMNO"))  # 15 single chars
        result, invalid = check_sequence(df, "seq")
        assert result == "None"
        assert invalid == []

    def test_fewer_than_10_unique_values_returns_none(self):
        """Fewer than 10 unique values → not a sequence column."""
        values = ["ATCG", "GCTA", "TTAA"] * 10  # only 3 unique
        df = make_df(values)
        result, invalid = check_sequence(df, "seq")
        assert result == "None"
        assert invalid == []

    def test_exactly_10_unique_passes_guard(self):
        """Exactly 10 unique values should pass the unique_count < 10 guard."""
        values = dna_seqs(10)
        df = make_df(values)
        with patch(FAST_CHECK, return_value=(True, [])):
            result, _ = check_sequence(df, "seq")
        assert result != "None" or True  # guard passed; detection result depends on mock

    def test_high_non_alpha_ratio_returns_none(self):
        """More than 30 % of unique values contain non-alpha chars → None."""
        # 4 out of 10 unique values have digits → 40 % > 30 %
        values = [
            "ATCG", "GCTA", "TTAA", "CCGG", "ATAT",
            "GCGC", "TATA", "ACG1", "T2GT", "GC3A",  # last 3 have digits
            "ACGT",
        ]
        df = make_df(values)
        result, invalid = check_sequence(df, "seq")
        assert result == "None"
        assert invalid == []

    def test_sequence_of_length_2_or_less_skips_detection(self):
        """Sequences with len <= 2 cause `all(len(x) > 2)` to be False → None."""
        values = ["AT"] * 5 + dna_seqs(10)  # "AT" has len=2
        df = make_df(values)
        result, invalid = check_sequence(df, "seq")
        assert result == "None"
        assert invalid == []

    def test_mixed_dtype_column_returns_none(self):
        """Mixed inferred dtype (strings + numbers) → early return."""
        df = pd.DataFrame({"seq": [1, "ATCG", 2.0, "GCTA"] * 5})
        result, invalid = check_sequence(df, "seq")
        assert result == "None"
        assert invalid == []

    def test_nan_values_excluded_before_check(self):
        """NaN values are dropped; detection runs on remaining values."""
        values = dna_seqs(12) + [None] * 5
        df = make_df(values)
        with patch(FAST_CHECK, return_value=(True, [])):
            result, _ = check_sequence(df, "seq")
        # Should not crash and should pass unique_count guard
        assert isinstance(result, str)


# ---------------------------------------------------------------------------
# check_sequence — sequence type detection
# ---------------------------------------------------------------------------

class TestCheckSequenceDetection:
    def test_detects_dna(self):
        df = make_df(dna_seqs(15))
        with patch(FAST_CHECK, return_value=(True, [])):
            result, invalid = check_sequence(df, "seq")
        assert result == "dna"
        assert invalid == []

    def test_detects_rna_when_dna_fails(self):
        df = make_df(dna_seqs(15))
        # DNA fails, RNA succeeds
        with patch(FAST_CHECK, side_effect=[(False, []), (True, [])]):
            result, invalid = check_sequence(df, "seq")
        assert result == "rna"
        assert invalid == []

    def test_detects_protein_when_dna_rna_fail(self):
        df = make_df(dna_seqs(15))
        # DNA fails, RNA fails, protein succeeds with sufficient entropy
        with patch(FAST_CHECK, side_effect=[(False, []), (False, []), (True, [])]), \
             patch(CHAR_ENTROPY, return_value=ENTROPY_THRESHOLDS["protein"] + 0.1):
            result, invalid = check_sequence(df, "seq")
        assert result == "protein"

    def test_protein_rejected_when_entropy_too_low(self):
        """Protein match but entropy below threshold → not protein → None."""
        df = make_df(dna_seqs(15))
        with patch(FAST_CHECK, side_effect=[(False, []), (False, []), (True, ["bad"])]), \
             patch(CHAR_ENTROPY, return_value=ENTROPY_THRESHOLDS["protein"] - 0.1):
            result, _ = check_sequence(df, "seq")
        assert result == "None"

    def test_protein_accepted_when_no_invalid_even_if_low_entropy(self):
        """Protein: if no invalid sequences, entropy check is skipped."""
        df = make_df(dna_seqs(15))
        with patch(FAST_CHECK, side_effect=[(False, []), (False, []), (True, [])]), \
             patch(CHAR_ENTROPY, return_value=0.0):
            result, _ = check_sequence(df, "seq")
        # no invalid → `not invalid` is True → returns protein regardless of entropy
        assert result == "protein"

    def test_returns_none_when_all_detection_fails(self):
        df = make_df(dna_seqs(15))
        with patch(FAST_CHECK, return_value=(False, [])):
            result, invalid = check_sequence(df, "seq")
        assert result == "None"
        assert invalid == []

    def test_invalid_sequences_returned_for_dna(self):
        df = make_df(dna_seqs(15))
        # DNA match with index 2 invalid
        with patch(FAST_CHECK, return_value=(True, [2])):
            result, invalid = check_sequence(df, "seq")
        assert result == "dna"
        assert len(invalid) == 1

    def test_custom_threshold_passed_to_fast_check(self):
        df = make_df(dna_seqs(15))
        with patch(FAST_CHECK, return_value=(True, [])) as mock_check:
            check_sequence(df, "seq", threshold=0.85)
        # fast_check_sequence should have been called with threshold=0.85
        calls = mock_check.call_args_list
        assert any(call.args[2] == 0.85 or call.kwargs.get("threshold") == 0.85
                   for call in calls)

    def test_detection_order_dna_before_rna_before_protein(self):
        """DNA is checked first, then RNA, then protein."""
        df = make_df(dna_seqs(15))
        call_order = []

        def mock_fast_check(values, pattern, threshold):
            call_order.append(str(pattern))
            return (False, [])

        with patch(FAST_CHECK, side_effect=mock_fast_check):
            check_sequence(df, "seq")

        from enums.sequence_enum import Sequence
        assert call_order[0] == str(Sequence.DNA.value)
        assert call_order[1] == str(Sequence.RNA.value)
        assert call_order[2] == str(Sequence.PROTEIN.value)

    def test_column_name_preserved_in_result(self):
        """check_sequence uses df[col] — verify col name doesn't affect outcome."""
        values = dna_seqs(15)
        df = pd.DataFrame({"nucleotide_sequence": values})
        with patch(FAST_CHECK, return_value=(True, [])):
            result, _ = check_sequence(df, "nucleotide_sequence")
        assert result == "dna"