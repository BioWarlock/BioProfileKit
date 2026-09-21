import pandas as pd
import pytest
import sys
import os

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from biological.uniprot_swissprot import (
    uniprot_flags,
    is_uniprot_accession,
    is_uniprot_protein_name,
    build_uniprot_lookups,
    UniProtFlags,
)
# Note: diagnose_protein_name_mismatch is intentionally not tested — it is
# marked "#ToDo: remove after debuggin/testing" in the source and is dead
# debug-only code, not part of the supported public behavior.


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_lookups(
    accessions=None,
    protein_names=None,
    ec_numbers=None,
    gene_names=None,
    name_to_primary=None,
):
    protein_names = protein_names or set()
    return {
        "accessions": set(accessions or []),
        "protein_names": set(protein_names),
        "protein_names_lower": {n.lower() for n in protein_names},
        "name_to_primary": {k.lower(): v for k, v in (name_to_primary or {}).items()},
        "ec_numbers": set(ec_numbers or []),
        "gene_names": set(gene_names or []),
    }


def build_swissprot_df():
    """Mimics the shape produced by data_utils.remote_data._parse_uniprot_dat."""
    return pd.DataFrame({
        "accession": ["P12345", "Q67890", "P00001"],
        "protein_name": ["Hemoglobin subunit alpha", "Insulin", "Cytochrome c"],
        "alt_names": ["Alpha-globin", None, None],
        "short_names": [None, "INS", None],
        "ec_numbers": [None, None, "EC 1.9.3.1"],
        "gene_names": ["HBA1;HBA2", "INS", "CYC1"],
    })


# ---------------------------------------------------------------------------
# is_uniprot_accession
# ---------------------------------------------------------------------------

class TestIsUniprotAccession:
    def test_returns_none_below_threshold(self):
        col = pd.Series(["P12345"] * 5 + ["BAD"] * 5)
        lookups = make_lookups(accessions=["P12345"])
        result = is_uniprot_accession(col, lookups, threshold=0.8)
        assert result is None

    def test_returns_result_above_threshold(self):
        col = pd.Series(["P12345"] * 9 + ["BAD"])
        lookups = make_lookups(accessions=["P12345"])
        result = is_uniprot_accession(col, lookups, threshold=0.8)
        assert result is not None
        assert result["invalid_values"] == ["BAD"]

    def test_all_valid_has_no_invalid_values(self):
        col = pd.Series(["P12345"] * 10)
        lookups = make_lookups(accessions=["P12345"])
        result = is_uniprot_accession(col, lookups, threshold=0.8)
        assert result["invalid_values"] is None
        assert result["validity_rate"] == 1.0

    def test_whitespace_is_stripped_before_matching(self):
        col = pd.Series([" P12345 "] * 10)
        lookups = make_lookups(accessions=["P12345"])
        result = is_uniprot_accession(col, lookups, threshold=0.8)
        assert result is not None


# ---------------------------------------------------------------------------
# is_uniprot_protein_name
# ---------------------------------------------------------------------------

class TestIsUniprotProteinName:
    def test_exact_match_above_threshold(self):
        col = pd.Series(["Insulin"] * 10)
        lookups = make_lookups(protein_names=["Insulin"])
        result = is_uniprot_protein_name(col, lookups, threshold=0.8)
        assert result is not None
        assert result["validity_rate"] == 1.0

    def test_returns_none_below_threshold(self):
        col = pd.Series(["Insulin"] * 5 + ["Unknown protein"] * 5)
        lookups = make_lookups(protein_names=["Insulin"])
        result = is_uniprot_protein_name(col, lookups, threshold=0.8)
        assert result is None

    def test_falls_back_to_cleaned_name_with_parenthetical_suffix(self):
        """'Insulin (precursor)' should match via the cleaned/lowercased
        fallback path when the raw value doesn't match exactly."""
        col = pd.Series(["Insulin (precursor)"] * 10)
        lookups = make_lookups(protein_names=["Insulin"])
        result = is_uniprot_protein_name(col, lookups, threshold=0.8)
        assert result is not None

    def test_outdated_names_detected(self):
        col = pd.Series(["Old Name"] * 10)
        lookups = make_lookups(
            protein_names=["Old Name"],
            name_to_primary={"Old Name": "New Name"},
        )
        result = is_uniprot_protein_name(col, lookups, threshold=0.8)
        assert result["outdated_names"] == {"Old Name": "New Name"}

    def test_no_outdated_names_when_current(self):
        col = pd.Series(["Insulin"] * 10)
        lookups = make_lookups(
            protein_names=["Insulin"],
            name_to_primary={"Insulin": "Insulin"},
        )
        result = is_uniprot_protein_name(col, lookups, threshold=0.8)
        assert result["outdated_names"] is None


# ---------------------------------------------------------------------------
# uniprot_flags — orchestration / dispatch
# ---------------------------------------------------------------------------

class TestUniprotFlags:
    def test_numeric_column_is_not_uniprot(self):
        df = pd.DataFrame({"col": [1, 2, 3, 4, 5]})
        result = uniprot_flags(df, "col", make_lookups())
        assert isinstance(result, UniProtFlags)
        assert result.is_uniprot is False
        assert result.match_type is None

    def test_all_null_column_is_not_uniprot(self):
        df = pd.DataFrame({"col": [None, None, None]})
        result = uniprot_flags(df, "col", make_lookups())
        assert result.is_uniprot is False

    def test_below_min_unique_is_not_uniprot(self):
        """Fewer than min_unique distinct values (default 2) — not enough
        signal to classify as a UniProt column."""
        df = pd.DataFrame({"col": ["P12345"]})
        result = uniprot_flags(df, "col", make_lookups(accessions=["P12345"]), min_unique=2)
        assert result.is_uniprot is False

    def test_accession_match_type(self):
        df = pd.DataFrame({"col": ["P12345"] * 8 + ["Q67890"] * 2})
        result = uniprot_flags(df, "col", make_lookups(accessions=["P12345", "Q67890"]))
        assert result.is_uniprot is True
        assert result.match_type == "accession"

    def test_protein_name_match_type(self):
        df = pd.DataFrame({"col": ["Insulin"] * 8 + ["Cytochrome c"] * 2})
        result = uniprot_flags(df, "col", make_lookups(protein_names=["Insulin", "Cytochrome c"]))
        assert result.is_uniprot is True
        assert result.match_type == "protein_name"

    def test_ec_number_match_type(self):
        df = pd.DataFrame({"col": ["EC 1.9.3.1"] * 8 + ["EC 2.1.1.1"] * 2})
        result = uniprot_flags(df, "col", make_lookups(ec_numbers=["EC 1.9.3.1", "EC 2.1.1.1"]))
        assert result.is_uniprot is True
        assert result.match_type == "ec_number"

    def test_gene_name_match_type(self):
        df = pd.DataFrame({"col": ["HBA1"] * 8 + ["HBA2"] * 2})
        result = uniprot_flags(df, "col", make_lookups(gene_names=["HBA1", "HBA2"]))
        assert result.is_uniprot is True
        assert result.match_type == "gene_name"

    def test_no_match_returns_not_uniprot(self):
        df = pd.DataFrame({"col": [f"random_value_{i}" for i in range(10)]})
        result = uniprot_flags(df, "col", make_lookups())
        assert result.is_uniprot is False
        assert result.match_type is None

    def test_ec_invalid_values_reported(self):
        df = pd.DataFrame({"col": ["EC 1.9.3.1"] * 9 + ["EC 9.9.9.9"]})
        result = uniprot_flags(df, "col", make_lookups(ec_numbers=["EC 1.9.3.1"]))
        assert result.invalid_values == ["EC 9.9.9.9"]

    def test_accession_checked_before_protein_name(self):
        """A column matching both accessions and protein names should be
        classified via the accession path (checked first)."""
        df = pd.DataFrame({"col": ["P12345"] * 8 + ["Q67890"] * 2})
        lookups = make_lookups(
            accessions=["P12345", "Q67890"],
            protein_names=["P12345", "Q67890"],
        )
        result = uniprot_flags(df, "col", lookups)
        assert result.match_type == "accession"


# ---------------------------------------------------------------------------
# build_uniprot_lookups
# ---------------------------------------------------------------------------

class TestBuildUniprotLookups:
    def test_returns_expected_keys(self):
        result = build_uniprot_lookups(build_swissprot_df())
        assert set(result.keys()) == {
            "accessions", "protein_names", "protein_names_lower",
            "name_to_primary", "ec_numbers", "gene_names",
        }

    def test_accessions_collected(self):
        result = build_uniprot_lookups(build_swissprot_df())
        assert result["accessions"] == {"P12345", "Q67890", "P00001"}

    def test_primary_protein_names_included(self):
        result = build_uniprot_lookups(build_swissprot_df())
        assert "Hemoglobin subunit alpha" in result["protein_names"]
        assert "Insulin" in result["protein_names"]

    def test_alt_and_short_names_map_to_primary(self):
        result = build_uniprot_lookups(build_swissprot_df())
        assert result["name_to_primary"]["alpha-globin"] == "Hemoglobin subunit alpha"
        assert result["name_to_primary"]["ins"] == "Insulin"

    def test_protein_names_lower_is_lowercased(self):
        result = build_uniprot_lookups(build_swissprot_df())
        assert "insulin" in result["protein_names_lower"]

    def test_ec_numbers_split_and_collected(self):
        result = build_uniprot_lookups(build_swissprot_df())
        assert "EC 1.9.3.1" in result["ec_numbers"]

    def test_gene_names_split_on_semicolon(self):
        result = build_uniprot_lookups(build_swissprot_df())
        assert "HBA1" in result["gene_names"]
        assert "HBA2" in result["gene_names"]

    def test_end_to_end_lookups_usable_by_uniprot_flags(self):
        """Lookups built from a realistic reference DataFrame should work
        directly with uniprot_flags, without any manual adaptation."""
        lookups = build_uniprot_lookups(build_swissprot_df())
        df = pd.DataFrame({"col": ["HBA1"] * 8 + ["HBA2"] * 2})
        result = uniprot_flags(df, "col", lookups)
        assert result.is_uniprot is True
        assert result.match_type == "gene_name"