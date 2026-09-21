from types import SimpleNamespace

import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from quality_assessment.biological_quality import (
    _check_sequence_validity,
    _check_sequence_redundancy,
    _check_taxonomy_validity,
    _check_unit_validity,
    _check_uniprot_validity,
)


def make_col(**overrides):
    defaults = dict(
        name="col1", sequence=None, invalid_seqs=None, dna_rna_data=None,
        protein_data=None, taxonomy=None, measurement_data=None, uniprot=None,
    )
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


# ---------------------------------------------------------------------------
# _check_sequence_validity
# ---------------------------------------------------------------------------

class TestCheckSequenceValidity:
    def test_no_sequence_columns_passes(self):
        cols = [make_col(sequence=None)]
        result = _check_sequence_validity(cols)
        assert result.status == "pass"
        assert result.message == "No sequence columns present"
        assert result.detail_link is None

    def test_invalid_sequences_warn(self):
        col = make_col(sequence="dna", invalid_seqs=[(0, "XXXX"), (1, "YYYY")])
        result = _check_sequence_validity([col])
        assert result.status == "warn"
        assert "2 invalid" in result.message

    def test_no_invalid_sequences_passes(self):
        col = make_col(sequence="dna", invalid_seqs=None)
        result = _check_sequence_validity([col])
        assert result.status == "pass"
        assert result.message == "All sequences valid"

    def test_high_ambiguous_ratio_fails(self):
        data = SimpleNamespace(ambiguous_base_ratio=SimpleNamespace(mean=20.0), stop_codon_ratio=None)
        col = make_col(sequence="dna", dna_rna_data=data)
        result = _check_sequence_validity([col])
        assert result.status == "fail"
        assert "mean ambiguous" in result.message

    def test_moderate_ambiguous_ratio_warns(self):
        data = SimpleNamespace(ambiguous_base_ratio=SimpleNamespace(mean=7.0), stop_codon_ratio=None)
        col = make_col(sequence="dna", dna_rna_data=data)
        result = _check_sequence_validity([col])
        assert result.status == "warn"

    def test_low_ambiguous_ratio_passes(self):
        data = SimpleNamespace(ambiguous_base_ratio=SimpleNamespace(mean=1.0), stop_codon_ratio=None)
        col = make_col(sequence="dna", dna_rna_data=data)
        result = _check_sequence_validity([col])
        assert result.status == "pass"

    def test_protein_stop_codon_ratio_warns(self):
        data = SimpleNamespace(ambiguous_residue_ratio=SimpleNamespace(mean=0.0), stop_codon_ratio=5.0)
        col = make_col(sequence="protein", protein_data=data)
        result = _check_sequence_validity([col])
        assert result.status == "warn"
        assert "with stop codons" in result.message

    def test_worst_status_wins_across_multiple_columns(self):
        col_warn = make_col(name="col_a", sequence="dna", invalid_seqs=[(0, "X")])
        data_fail = SimpleNamespace(ambiguous_base_ratio=SimpleNamespace(mean=50.0), stop_codon_ratio=None)
        col_fail = make_col(name="col_b", sequence="dna", dna_rna_data=data_fail)
        result = _check_sequence_validity([col_warn, col_fail])
        assert result.status == "fail"

    def test_detail_link_set_when_sequence_columns_exist(self):
        col = make_col(sequence="protein")
        result = _check_sequence_validity([col])
        assert result.detail_link == "#columns"


# ---------------------------------------------------------------------------
# _check_sequence_redundancy
# ---------------------------------------------------------------------------

class TestCheckSequenceRedundancy:
    def test_no_dna_columns_passes(self):
        cols = [make_col(sequence="protein")]
        result = _check_sequence_redundancy(cols)
        assert result.status == "pass"
        assert result.message == "No DNA/RNA columns present"

    def test_no_significant_redundancy_passes(self):
        data = SimpleNamespace(reverse_complement_ratio=2.0)
        col = make_col(sequence="dna", dna_rna_data=data)
        result = _check_sequence_redundancy([col])
        assert result.status == "pass"

    def test_moderate_redundancy_warns(self):
        data = SimpleNamespace(reverse_complement_ratio=15.0)
        col = make_col(sequence="dna", dna_rna_data=data)
        result = _check_sequence_redundancy([col])
        assert result.status == "warn"

    def test_high_redundancy_fails(self):
        data = SimpleNamespace(reverse_complement_ratio=40.0)
        col = make_col(sequence="dna", dna_rna_data=data)
        result = _check_sequence_redundancy([col])
        assert result.status == "fail"

    def test_missing_dna_rna_data_skipped_safely(self):
        col = make_col(sequence="dna", dna_rna_data=None)
        result = _check_sequence_redundancy([col])
        assert result.status == "pass"


# ---------------------------------------------------------------------------
# _check_taxonomy_validity
# ---------------------------------------------------------------------------

class TestCheckTaxonomyValidity:
    def test_no_taxonomy_columns_passes(self):
        cols = [make_col(taxonomy=None)]
        result = _check_taxonomy_validity(cols)
        assert result.status == "pass"
        assert result.message == "No taxonomy columns present"

    def test_invalid_names_warn(self):
        tax = SimpleNamespace(invalid_names=["Bad Name 1", "Bad Name 2"])
        col = make_col(taxonomy=tax)
        result = _check_taxonomy_validity([col])
        assert result.status == "warn"
        assert "2 invalid" in result.message

    def test_no_invalid_names_passes(self):
        tax = SimpleNamespace(invalid_names=None)
        col = make_col(taxonomy=tax)
        result = _check_taxonomy_validity([col])
        assert result.status == "pass"
        assert result.message == "Taxonomy valid"


# ---------------------------------------------------------------------------
# _check_unit_validity
# ---------------------------------------------------------------------------

class TestCheckUnitValidity:
    def test_no_unit_columns_passes(self):
        cols = [make_col(measurement_data=None)]
        result = _check_unit_validity(cols)
        assert result.status == "pass"
        assert result.message == "No unit columns present"

    def test_multiple_units_warns(self):
        m = SimpleNamespace(unit_counts={"mg": 5, "g": 2}, with_measurement=False)
        col = make_col(measurement_data=m)
        result = _check_unit_validity([col])
        assert result.status == "warn"
        assert "2 units found" in result.message

    def test_single_unit_no_measurement_passes(self):
        m = SimpleNamespace(unit_counts={"mg": 5}, with_measurement=False)
        col = make_col(measurement_data=m)
        result = _check_unit_validity([col])
        assert result.status == "pass"

    def test_with_measurement_warns(self):
        m = SimpleNamespace(unit_counts={"mg": 5}, with_measurement=True)
        col = make_col(measurement_data=m)
        result = _check_unit_validity([col])
        assert result.status == "warn"
        assert "measurement with unit found" in result.message

    def test_both_conditions_produce_two_notes(self):
        m = SimpleNamespace(unit_counts={"mg": 5, "g": 2}, with_measurement=True)
        col = make_col(measurement_data=m)
        result = _check_unit_validity([col])
        assert result.status == "warn"
        assert "units found" in result.message
        assert "measurement with unit found" in result.message


# ---------------------------------------------------------------------------
# _check_uniprot_validity
# ---------------------------------------------------------------------------

class TestCheckUniprotValidity:
    def test_no_uniprot_columns_passes(self):
        cols = [make_col(uniprot=None)]
        result = _check_uniprot_validity(cols)
        assert result.status == "pass"
        assert result.message == "No UniProt-matched columns present"

    def test_non_matched_uniprot_column_ignored(self):
        uni = SimpleNamespace(is_uniprot=False, invalid_values=None, outdated_names=None, match_type=None)
        col = make_col(uniprot=uni)
        result = _check_uniprot_validity([col])
        assert result.status == "pass"

    def test_invalid_values_warn(self):
        uni = SimpleNamespace(is_uniprot=True, invalid_values=["BAD1"], outdated_names=None, match_type="accession")
        col = make_col(uniprot=uni)
        result = _check_uniprot_validity([col])
        assert result.status == "warn"
        assert "1 invalid" in result.message
        assert "accession" in result.message

    def test_outdated_names_warn(self):
        uni = SimpleNamespace(is_uniprot=True, invalid_values=None, outdated_names={"Old": "New"}, match_type="protein_name")
        col = make_col(uniprot=uni)
        result = _check_uniprot_validity([col])
        assert result.status == "warn"
        assert "1 outdated name" in result.message

    def test_clean_match_passes(self):
        uni = SimpleNamespace(is_uniprot=True, invalid_values=None, outdated_names=None, match_type="accession")
        col = make_col(uniprot=uni)
        result = _check_uniprot_validity([col])
        assert result.status == "pass"
        assert result.message == "UniProt-matched columns valid"