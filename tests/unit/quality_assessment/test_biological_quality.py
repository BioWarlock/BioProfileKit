"""Unit tests for quality_assessment/biological_quality.py"""
import sys
import os
import pytest
from unittest.mock import MagicMock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from quality_assessment.biological_quality import (
    _check_sequence_validity,
    _check_sequence_redundancy,
    _check_taxonomy_validity,
    _check_unit_validity,
)


def _seq_col(name, sequence, invalid=None, dna_rna_data=None, protein_data=None):
    c = MagicMock()
    c.name = name
    c.sequence = sequence
    c.invalid_seqs = invalid or []
    c.dna_rna_data = dna_rna_data
    c.protein_data = protein_data
    return c


class TestCheckSequenceValidity:
    def test_no_seq_cols_passes(self):
        col = MagicMock(); col.sequence = "None"
        result = _check_sequence_validity([col])
        assert result.status == "pass"
        assert "No sequence columns" in result.message

    def test_invalid_seqs_triggers_warn(self):
        col = _seq_col("seq", "dna", invalid=["BAD1", "BAD2"])
        result = _check_sequence_validity([col])
        assert result.status == "warn"
        assert "2 invalid" in result.message

    def test_high_ambiguous_triggers_fail(self):
        col = _seq_col("seq", "protein")
        data = MagicMock()
        data.ambiguous_base_ratio = None
        data.ambiguous_residue_ratio = MagicMock(mean=20.0)  # > FAIL=15
        data.stop_codon_ratio = None
        col.protein_data = data
        col.dna_rna_data = None
        result = _check_sequence_validity([col])
        assert result.status == "fail"

    def test_medium_ambiguous_triggers_warn(self):
        col = _seq_col("seq", "protein")
        data = MagicMock()
        data.ambiguous_base_ratio = None
        data.ambiguous_residue_ratio = MagicMock(mean=8.0)  # > WARN=5 < FAIL=15
        data.stop_codon_ratio = None
        col.protein_data = data
        col.dna_rna_data = None
        result = _check_sequence_validity([col])
        assert result.status == "warn"

    def test_stop_codon_warn(self):
        col = _seq_col("seq", "protein")
        data = MagicMock()
        data.ambiguous_base_ratio = None
        data.ambiguous_residue_ratio = MagicMock(mean=0.0)
        data.stop_codon_ratio = 5.0   # > STOP_CODON_WARN=1
        col.protein_data = data
        col.dna_rna_data = None
        result = _check_sequence_validity([col])
        assert result.status == "warn"
        assert "stop codons" in result.message

    def test_all_valid_passes(self):
        col = _seq_col("seq", "dna")
        data = MagicMock()
        data.ambiguous_base_ratio = MagicMock(mean=0.0)
        data.ambiguous_residue_ratio = None
        data.stop_codon_ratio = None
        col.dna_rna_data = data
        col.protein_data = None
        result = _check_sequence_validity([col])
        assert result.status == "pass"

    def test_no_data_skips_metric_checks(self):
        col = _seq_col("seq", "dna")
        col.dna_rna_data = None
        col.protein_data = None
        result = _check_sequence_validity([col])
        assert result.status == "pass"


class TestCheckSequenceRedundancy:
    def test_no_dna_cols_passes(self):
        col = MagicMock(); col.sequence = "protein"
        result = _check_sequence_redundancy([col])
        assert result.status == "pass"
        assert "No DNA/RNA" in result.message

    def test_high_rc_fails(self):
        col = MagicMock(); col.sequence = "dna"
        data = MagicMock(); data.reverse_complement_ratio = 35.0  # > FAIL=30
        col.dna_rna_data = data
        result = _check_sequence_redundancy([col])
        assert result.status == "fail"

    def test_medium_rc_warns(self):
        col = MagicMock(); col.sequence = "dna"
        data = MagicMock(); data.reverse_complement_ratio = 15.0  # > WARN=10 < FAIL=30
        col.dna_rna_data = data
        result = _check_sequence_redundancy([col])
        assert result.status == "warn"

    def test_low_rc_passes(self):
        col = MagicMock(); col.sequence = "dna"
        data = MagicMock(); data.reverse_complement_ratio = 2.0
        col.dna_rna_data = data
        result = _check_sequence_redundancy([col])
        assert result.status == "pass"

    def test_no_data_skips(self):
        col = MagicMock(); col.sequence = "dna"
        col.dna_rna_data = None
        result = _check_sequence_redundancy([col])
        assert result.status == "pass"

    def test_no_rc_attribute_skips(self):
        col = MagicMock(); col.sequence = "dna"
        data = MagicMock(); data.reverse_complement_ratio = None
        col.dna_rna_data = data
        result = _check_sequence_redundancy([col])
        assert result.status == "pass"


class TestCheckTaxonomyValidity:
    def test_no_taxonomy_passes(self):
        col = MagicMock(); col.taxonomy = None
        result = _check_taxonomy_validity([col])
        assert result.status == "pass"
        assert "No taxonomy" in result.message

    def test_invalid_names_warns(self):
        col = MagicMock()
        col.taxonomy = MagicMock(invalid_names=["BadName1", "BadName2"])
        result = _check_taxonomy_validity([col])
        assert result.status == "warn"
        assert "2 invalid" in result.message

    def test_no_invalid_passes(self):
        col = MagicMock()
        col.taxonomy = MagicMock(invalid_names=[])
        result = _check_taxonomy_validity([col])
        assert result.status == "pass"
        assert "valid" in result.message.lower()

    def test_none_invalid_passes(self):
        col = MagicMock()
        col.taxonomy = MagicMock(invalid_names=None)
        result = _check_taxonomy_validity([col])
        assert result.status == "pass"


class TestCheckUnitValidity:
    def test_no_unit_cols_passes(self):
        col = MagicMock(); col.measurement_data = None
        result = _check_unit_validity([col])
        assert result.status == "pass"
        assert "No unit" in result.message

    def test_multiple_units_warns(self):
        col = MagicMock()
        col.measurement_data = MagicMock(
            unit_counts={"mg/L": 5, "g/L": 3},
            with_measurement=False
        )
        result = _check_unit_validity([col])
        assert result.status == "warn"
        assert "2 units" in result.message

    def test_with_measurement_warns(self):
        col = MagicMock()
        col.measurement_data = MagicMock(
            unit_counts={"mg/L": 5},
            with_measurement=True
        )
        result = _check_unit_validity([col])
        assert result.status == "warn"
        assert "measurement with unit" in result.message

    def test_single_unit_no_measurement_passes(self):
        col = MagicMock()
        col.measurement_data = MagicMock(
            unit_counts={"mg/L": 10},
            with_measurement=False
        )
        result = _check_unit_validity([col])
        assert result.status == "pass"