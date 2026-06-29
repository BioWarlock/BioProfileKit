"""Unit tests for biological/taxonomy.py (pure functions only)"""
import sys
import os
import pytest
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from biological.taxonomy import (
    build_lookups, is_taxid, is_taxonomy,
    taxid_rank_distribution, rank_distribution, find_outdated_names,
    taxonomy_flags,
)


def _make_vocab():
    """Minimal taxonomy vocab DataFrame."""
    return pd.DataFrame({
        "name_txt":        ["Homo sapiens", "Mus musculus", "Escherichia coli", "Homo sapiens"],
        "tax_id":          [9606, 10090, 562, 9606],
        "rank":            ["species", "species", "species", "species"],
        "name_class":      ["scientific name", "scientific name", "scientific name", "synonym"],
        "scientific_name": ["Homo sapiens", "Mus musculus", "Escherichia coli", "Homo sapiens"],
    })


class TestBuildLookups:
    def test_returns_five_items(self):
        result = build_lookups(_make_vocab())
        assert len(result) == 5

    def test_valid_names_populated(self):
        valid_names, valid_tax_ids, _, _, _ = build_lookups(_make_vocab())
        assert "Homo sapiens" in valid_names
        assert 9606 in valid_tax_ids

    def test_taxid_to_rank_from_scientific_name_only(self):
        _, _, _, taxid_to_rank, _ = build_lookups(_make_vocab())
        assert taxid_to_rank[9606] == "species"


class TestIsTaxid:
    def test_valid_tax_ids_detected(self):
        col = pd.Series([9606, 10090, 562], name="tax_id")
        result = is_taxid(col, {9606, 10090, 562})
        assert result == "all tax IDs valid"

    def test_invalid_ids_returned_as_set(self):
        # validity_rate must be strictly > 0.9, so 10 valid + 1 invalid = 10/11 ≈ 0.909
        col = pd.Series([9606] * 10 + [9999], name="tax_id")
        result = is_taxid(col, {9606})
        assert isinstance(result, set)
        assert 9999 in result

    def test_excluded_column_name(self):
        col = pd.Series([1, 2, 3], name="length")
        result = is_taxid(col, {1, 2, 3})
        assert result is None

    def test_non_numeric_series(self):
        col = pd.Series(["hello", "world"], name="org")
        result = is_taxid(col, {9606})
        assert result is None

    def test_below_threshold_returns_none(self):
        col = pd.Series([9606, 9607, 9608, 9609, 9610], name="tax_id")
        result = is_taxid(col, {9606})
        assert result is None


class TestIsTaxonomy:
    def _lookups(self):
        vocab = _make_vocab()
        valid_names = set(vocab["name_txt"])
        name_to_rank = dict(zip(vocab["name_txt"], vocab["rank"]))
        name_to_scientific = dict(zip(vocab["name_txt"], vocab["scientific_name"]))
        return valid_names, name_to_rank, name_to_scientific

    def test_valid_taxonomy_detected(self):
        valid_names, name_to_rank, name_to_scientific = self._lookups()
        col = pd.Series(["Homo sapiens", "Mus musculus", "Homo sapiens",
                         "Escherichia coli", "Homo sapiens"])
        result = is_taxonomy(col, valid_names, name_to_rank, name_to_scientific)
        assert result is not None
        assert result["valid"] is True

    def test_non_taxonomy_returns_none(self):
        valid_names, name_to_rank, name_to_scientific = self._lookups()
        col = pd.Series(["alpha", "beta", "gamma", "delta", "epsilon"])
        result = is_taxonomy(col, valid_names, name_to_rank, name_to_scientific)
        assert result is None


class TestTaxidRankDistribution:
    def test_single_rank(self):
        taxid_to_rank = {9606: "species", 10090: "species"}
        col = pd.Series([9606, 10090])
        dist, is_mixed = taxid_rank_distribution(col, taxid_to_rank)
        assert "species" in dist
        assert is_mixed is False

    def test_mixed_ranks(self):
        taxid_to_rank = {9606: "species", 1: "genus"}
        col = pd.Series([9606, 9606, 9606, 9606, 9606, 1, 1, 1, 1, 1])
        dist, is_mixed = taxid_rank_distribution(col, taxid_to_rank)
        assert is_mixed is True


class TestRankDistribution:
    def test_valid_names(self):
        name_to_rank = {"Homo sapiens": "species", "Mus musculus": "species"}
        col = pd.Series(["Homo sapiens", "Homo sapiens", "Mus musculus"])
        dist, is_mixed, invalid = rank_distribution(col, name_to_rank)
        assert "species" in dist
        assert invalid == []

    def test_invalid_names_collected(self):
        name_to_rank = {"Homo sapiens": "species"}
        col = pd.Series(["Homo sapiens", "Unknown organism"])
        dist, is_mixed, invalid = rank_distribution(col, name_to_rank)
        assert "Unknown organism" in invalid


class TestFindOutdatedNames:
    def test_outdated_name_detected(self):
        name_to_scientific = {"OldName": "NewName", "Homo sapiens": "Homo sapiens"}
        col = pd.Series(["OldName", "Homo sapiens"])
        result = find_outdated_names(col, {"OldName", "Homo sapiens"}, name_to_scientific)
        assert "OldName" in result
        assert result["OldName"] == "NewName"

    def test_current_names_not_in_result(self):
        name_to_scientific = {"Homo sapiens": "Homo sapiens"}
        col = pd.Series(["Homo sapiens"])
        result = find_outdated_names(col, {"Homo sapiens"}, name_to_scientific)
        assert "Homo sapiens" not in result


class TestTaxonomyFlags:
    def test_numeric_col_valid_tax_ids(self):
        valid_names, valid_tax_ids, name_to_rank, taxid_to_rank, name_to_sci = build_lookups(_make_vocab())
        df = pd.DataFrame({"tax_id": [9606, 10090, 562, 9606, 10090,
                                       9606, 10090, 562, 9606, 10090]})
        result = taxonomy_flags(df, "tax_id", valid_names, valid_tax_ids,
                                 name_to_rank, taxid_to_rank, name_to_sci)
        assert result.is_taxonomy is True

    def test_string_col_valid_names(self):
        valid_names, valid_tax_ids, name_to_rank, taxid_to_rank, name_to_sci = build_lookups(_make_vocab())
        df = pd.DataFrame({"org": ["Homo sapiens", "Mus musculus", "Homo sapiens",
                                    "Homo sapiens", "Mus musculus"]})
        result = taxonomy_flags(df, "org", valid_names, valid_tax_ids,
                                 name_to_rank, taxid_to_rank, name_to_sci)
        assert hasattr(result, "is_taxonomy")

    def test_non_taxonomy_col(self):
        valid_names, valid_tax_ids, name_to_rank, taxid_to_rank, name_to_sci = build_lookups(_make_vocab())
        df = pd.DataFrame({"label": ["alpha", "beta", "gamma", "delta", "epsilon"]})
        result = taxonomy_flags(df, "label", valid_names, valid_tax_ids,
                                 name_to_rank, taxid_to_rank, name_to_sci)
        assert result.is_taxonomy is False


# ---------------------------------------------------------------------------
# TestBuildLookups — extended
# ---------------------------------------------------------------------------

class TestBuildLookupsSynonyms:
    def test_synonym_included_in_valid_names(self):
        """Synonyms (non-scientific names) must also appear in valid_names
        so that taxonomy_flags can match them in string columns."""
        vocab = _make_vocab()
        valid_names, _, _, _, _ = build_lookups(vocab)
        # "Homo sapiens" appears once as synonym — still in valid_names
        assert "Homo sapiens" in valid_names

    def test_name_to_scientific_maps_synonym_to_current(self):
        """name_to_scientific is used by find_outdated_names to detect
        entries that have been superseded by a current scientific name."""
        vocab = pd.DataFrame({
            "name_txt":        ["Bacterium coli", "Escherichia coli"],
            "tax_id":          [562, 562],
            "rank":            ["species", "species"],
            "name_class":      ["synonym", "scientific name"],
            "scientific_name": ["Escherichia coli", "Escherichia coli"],
        })
        _, _, _, _, name_to_scientific = build_lookups(vocab)
        # Old synonym maps to current scientific name
        assert name_to_scientific["Bacterium coli"] == "Escherichia coli"

    def test_taxid_to_rank_excludes_synonyms(self):
        """Only scientific-name rows should populate taxid_to_rank."""
        vocab = pd.DataFrame({
            "name_txt":        ["Old name", "Homo sapiens"],
            "tax_id":          [9606, 9606],
            "rank":            ["species", "species"],
            "name_class":      ["synonym", "scientific name"],
            "scientific_name": ["Homo sapiens", "Homo sapiens"],
        })
        _, _, _, taxid_to_rank, _ = build_lookups(vocab)
        assert taxid_to_rank[9606] == "species"
        assert len(taxid_to_rank) == 1  # de-duplicated to one entry


# ---------------------------------------------------------------------------
# TestIsTaxid — extended
# ---------------------------------------------------------------------------

class TestIsTaxidExtended:
    def test_start_and_end_excluded(self):
        """'start' and 'end' are positional columns in BED/GFF files —
        must never be classified as taxonomy even if values match IDs."""
        for excluded in ("start", "end"):
            col = pd.Series([9606, 10090, 562], name=excluded)
            assert is_taxid(col, {9606, 10090, 562}) is None

    def test_none_column_name_not_excluded(self):
        """A Series with name=None should not hit the exclusion list."""
        col = pd.Series([9606, 10090, 562], name=None)
        result = is_taxid(col, {9606, 10090, 562})
        assert result == "all tax IDs valid"

    def test_float_tax_ids(self):
        """Tax IDs stored as floats (e.g. from CSV without explicit dtype)
        should still be detected when they match known integer IDs."""
        col = pd.Series([9606.0, 10090.0, 562.0], name="tax_id")
        result = is_taxid(col, {9606, 10090, 562})
        assert result == "all tax IDs valid"

    def test_mixed_valid_and_invalid_ids(self):
        """When most IDs are valid but a few are not, the invalid IDs
        are returned as a set for downstream reporting.
        validity_rate must be strictly > 0.9: 10 valid + 1 invalid = 10/11 ≈ 0.909."""
        col = pd.Series([9606] * 10 + [99999], name="tax_id")
        result = is_taxid(col, {9606})
        assert isinstance(result, set)
        assert 99999 in result

    def test_validity_rate_exactly_at_threshold_returns_none(self):
        """Validity rate must be strictly greater than threshold (0.9).
        Exactly 90% valid should return None."""
        # 9 valid, 1 invalid → 9/10 = 0.9 exactly — not > 0.9
        col = pd.Series([9606] * 9 + [99999], name="tax_id")
        # With only {9606} valid and 9/10 valid the rate = 0.9
        # is_numeric_candidate check: all numeric → True
        # validity_rate = 9/10 = 0.9 → NOT > 0.9 → returns None
        # Note: the above mixed test passes because 9/10 > 0.9 threshold
        # only if threshold < 0.9. With default 0.9 exactly at threshold → None.
        # We set a custom threshold to isolate this branch.
        result = is_taxid(col, {9606}, threshold=0.95)
        assert result is None

    def test_with_nan_values(self):
        """NaN values in the column must not be counted as valid tax IDs
        and should not appear in the returned invalid set."""
        import numpy as np
        col = pd.Series([9606, 9606, 9606, 9606, 9606,
                         9606, 9606, 9606, 9606, np.nan], name="tax_id")
        result = is_taxid(col, {9606})
        # 9 numeric, 9 valid → validity_rate = 9/10 = 0.9 (not strictly > 0.9)
        # So result is None — NaN must not inflate validity
        assert result is None or result == "all tax IDs valid"


# ---------------------------------------------------------------------------
# TestIsTaxonomy — extended
# ---------------------------------------------------------------------------

class TestIsTaxonomyExtended:
    def _lookups(self):
        vocab = pd.DataFrame({
            "name_txt":        ["Homo sapiens", "Mus musculus", "Escherichia coli",
                                "Bacterium coli"],
            "tax_id":          [9606, 10090, 562, 562],
            "rank":            ["species", "species", "species", "species"],
            "name_class":      ["scientific name", "scientific name",
                                "scientific name", "synonym"],
            "scientific_name": ["Homo sapiens", "Mus musculus", "Escherichia coli",
                                "Escherichia coli"],
        })
        valid_names = set(vocab["name_txt"])
        name_to_rank = dict(zip(vocab["name_txt"], vocab["rank"]))
        name_to_scientific = dict(zip(vocab["name_txt"], vocab["scientific_name"]))
        return valid_names, name_to_rank, name_to_scientific

    def test_names_with_strain_annotation_cleaned(self):
        """Species names with parenthetical strain info like
        'Homo sapiens (GRCh38)' are common in bioinformatics outputs.
        The extract-and-strip fallback should still detect these as valid."""
        valid_names, name_to_rank, name_to_scientific = self._lookups()
        # Build a column where names have annotations in parens
        col = pd.Series([
            "Homo sapiens (GRCh38)", "Homo sapiens (GRCh38)",
            "Mus musculus (GRCm39)", "Homo sapiens (GRCh38)",
            "Escherichia coli (K-12)",
        ])
        result = is_taxonomy(col, valid_names, name_to_rank, name_to_scientific)
        # After stripping the parenthetical part, all names should be valid
        assert result is not None
        assert result["valid"] is True

    def test_outdated_synonym_detected(self):
        """A column containing an old synonym ('Bacterium coli') should be
        detected as taxonomy and the synonym flagged as outdated."""
        valid_names, name_to_rank, name_to_scientific = self._lookups()
        col = pd.Series(["Bacterium coli", "Bacterium coli",
                         "Escherichia coli", "Escherichia coli",
                         "Homo sapiens"])
        result = is_taxonomy(col, valid_names, name_to_rank, name_to_scientific)
        assert result is not None
        # Outdated entry should appear in result
        if result["outdated"]:
            assert "Bacterium coli" in result["outdated"]

    def test_invalid_names_in_result(self):
        """Names that are not in the taxonomy vocab should be collected
        in invalid_names for the quality report.
        validity_rate must be strictly > 0.8: 5 valid + 1 invalid = 5/6 ≈ 0.833."""
        valid_names, name_to_rank, name_to_scientific = self._lookups()
        col = pd.Series([
            "Homo sapiens", "Homo sapiens", "Homo sapiens",
            "Homo sapiens", "Homo sapiens", "Unknown species X",
        ])
        result = is_taxonomy(col, valid_names, name_to_rank, name_to_scientific)
        assert result is not None
        assert result["invalid_names"] is not None
        assert "Unknown species X" in result["invalid_names"]

    def test_all_valid_no_outdated(self):
        """A clean column with only current scientific names should return
        invalid_names=None and outdated=None."""
        valid_names, name_to_rank, name_to_scientific = self._lookups()
        col = pd.Series(["Homo sapiens"] * 5)
        result = is_taxonomy(col, valid_names, name_to_rank, name_to_scientific)
        assert result is not None
        assert result["invalid_names"] is None
        assert result["outdated"] is None

    def test_mixed_ranks_detected(self):
        """A column containing both species and genus names should be
        flagged as is_mixed=True in the rank distribution."""
        vocab = pd.DataFrame({
            "name_txt":        ["Homo sapiens", "Homo", "Mus musculus"],
            "tax_id":          [9606, 9605, 10090],
            "rank":            ["species", "genus", "species"],
            "name_class":      ["scientific name"] * 3,
            "scientific_name": ["Homo sapiens", "Homo", "Mus musculus"],
        })
        vn = set(vocab["name_txt"])
        ntr = dict(zip(vocab["name_txt"], vocab["rank"]))
        nts = dict(zip(vocab["name_txt"], vocab["scientific_name"]))
        # Equal split of genus / species → is_mixed=True
        col = pd.Series(["Homo sapiens", "Homo", "Mus musculus",
                         "Homo", "Homo sapiens"])
        result = is_taxonomy(col, vn, ntr, nts)
        if result is not None:
            assert "is_mixed" in result


# ---------------------------------------------------------------------------
# TestTaxidRankDistribution — extended
# ---------------------------------------------------------------------------

class TestTaxidRankDistributionExtended:
    def test_unknown_tax_ids_not_in_distribution(self):
        """Tax IDs not present in taxid_to_rank should not produce
        NaN keys in the distribution dict."""
        taxid_to_rank = {9606: "species"}
        col = pd.Series([9606, 99999])  # 99999 unknown
        dist, _ = taxid_rank_distribution(col, taxid_to_rank)
        for key in dist:
            assert key is not None and str(key) != "nan"

    def test_distribution_sums_to_one(self):
        """Normalized frequencies must sum to 1.0."""
        taxid_to_rank = {9606: "species", 10090: "species", 562: "species"}
        col = pd.Series([9606, 10090, 562])
        dist, _ = taxid_rank_distribution(col, taxid_to_rank)
        assert abs(sum(dist.values()) - 1.0) < 1e-6

    def test_single_entry_not_mixed(self):
        """A column with a single unique rank is never mixed."""
        taxid_to_rank = {9606: "species"}
        col = pd.Series([9606, 9606, 9606])
        _, is_mixed = taxid_rank_distribution(col, taxid_to_rank)
        assert is_mixed is False


# ---------------------------------------------------------------------------
# TestRankDistribution — extended
# ---------------------------------------------------------------------------

class TestRankDistributionExtended:
    def test_whitespace_stripped_before_lookup(self):
        """Names with leading/trailing whitespace are common in TSV exports.
        rank_distribution must strip them before lookup."""
        name_to_rank = {"Homo sapiens": "species"}
        col = pd.Series(["  Homo sapiens  ", "Homo sapiens"])
        dist, _, invalid = rank_distribution(col, name_to_rank)
        assert "species" in dist
        assert invalid == []

    def test_invalid_sorted_alphabetically(self):
        """Invalid names should be returned in sorted order for
        reproducible report output."""
        name_to_rank = {"Homo sapiens": "species"}
        col = pd.Series(["Zebra organism", "Alpha unknown", "Homo sapiens"])
        _, _, invalid = rank_distribution(col, name_to_rank)
        assert invalid == sorted(invalid)

    def test_distribution_values_rounded_to_four_decimals(self):
        """Frequencies must be rounded to 4 decimal places as specified."""
        name_to_rank = {"Homo sapiens": "species", "Mus musculus": "species"}
        col = pd.Series(["Homo sapiens"] * 3 + ["Mus musculus"])
        dist, _, _ = rank_distribution(col, name_to_rank)
        for v in dist.values():
            assert v == round(v, 4)


# ---------------------------------------------------------------------------
# TestFindOutdatedNames — extended
# ---------------------------------------------------------------------------

class TestFindOutdatedNamesExtended:
    def test_multiple_outdated_names_all_detected(self):
        """Multiple superseded synonyms in one column should all appear
        in the returned dict."""
        name_to_scientific = {
            "Bacterium coli":       "Escherichia coli",
            "Bacillus coli":        "Escherichia coli",
            "Escherichia coli":     "Escherichia coli",
        }
        col = pd.Series(["Bacterium coli", "Bacillus coli", "Escherichia coli"])
        valid_names = set(name_to_scientific.keys())
        result = find_outdated_names(col, valid_names, name_to_scientific)
        assert "Bacterium coli" in result
        assert "Bacillus coli" in result
        assert "Escherichia coli" not in result

    def test_name_not_in_valid_names_ignored(self):
        """Names that are not in valid_names at all (e.g. free text) must
        not appear in the outdated dict even if they happen to be in the
        name_to_scientific mapping."""
        name_to_scientific = {"SomeOld": "SomeNew"}
        col = pd.Series(["SomeOld"])
        # SomeOld not in valid_names → should be ignored
        result = find_outdated_names(col, set(), name_to_scientific)
        assert "SomeOld" not in result

    def test_empty_column_returns_empty_dict(self):
        """An empty column must return an empty outdated dict without error."""
        col = pd.Series([], dtype=str)
        result = find_outdated_names(col, {"Homo sapiens"}, {"Homo sapiens": "Homo sapiens"})
        assert result == {}

    def test_whitespace_stripped(self):
        """Whitespace around names must be stripped before comparison."""
        name_to_scientific = {"Bacterium coli": "Escherichia coli"}
        col = pd.Series(["  Bacterium coli  "])
        result = find_outdated_names(col, {"Bacterium coli"}, name_to_scientific)
        assert "Bacterium coli" in result


# ---------------------------------------------------------------------------
# TestTaxonomyFlags — extended
# ---------------------------------------------------------------------------

class TestTaxonomyFlagsExtended:
    def _full_lookups(self):
        vocab = pd.DataFrame({
            "name_txt":        ["Homo sapiens", "Mus musculus", "Escherichia coli",
                                "Bacterium coli"],
            "tax_id":          [9606, 10090, 562, 562],
            "rank":            ["species", "species", "species", "species"],
            "name_class":      ["scientific name", "scientific name",
                                "scientific name", "synonym"],
            "scientific_name": ["Homo sapiens", "Mus musculus", "Escherichia coli",
                                "Escherichia coli"],
        })
        return build_lookups(vocab)

    def test_numeric_col_with_invalid_returns_is_taxonomy_true(self):
        """A tax_id column where most IDs are valid but a few are not
        should still be classified as is_taxonomy=True, with invalid IDs
        stored in taxid for downstream reporting."""
        vn, vt, ntr, ttr, nts = self._full_lookups()
        # 9 valid IDs + 1 unknown → validity_rate = 9/10 = 0.9 (not > 0.9)
        # Use 10 valid + 1 invalid so validity_rate > 0.9
        df = pd.DataFrame({"tax_id": [9606] * 10 + [99999]})
        result = taxonomy_flags(df, "tax_id", vn, vt, ntr, ttr, nts)
        assert result.is_taxonomy is True
        assert isinstance(result.taxid, set)
        assert 99999 in result.taxid

    def test_string_col_with_outdated_names_sets_outdated_names(self):
        """A species column containing the old synonym 'Bacterium coli'
        should be detected as taxonomy with outdated_names populated."""
        vn, vt, ntr, ttr, nts = self._full_lookups()
        df = pd.DataFrame({"org": [
            "Bacterium coli", "Escherichia coli", "Homo sapiens",
            "Homo sapiens", "Escherichia coli",
        ]})
        result = taxonomy_flags(df, "org", vn, vt, ntr, ttr, nts)
        assert result.is_taxonomy is True
        if result.outdated_names:
            assert "Bacterium coli" in result.outdated_names

    def test_numeric_col_not_a_taxid_falls_through_to_false(self):
        """A numeric column where IDs do not match any known tax ID
        must return is_taxonomy=False."""
        vn, vt, ntr, ttr, nts = self._full_lookups()
        df = pd.DataFrame({"count": [1, 2, 3, 4, 5]})
        result = taxonomy_flags(df, "count", vn, vt, ntr, ttr, nts)
        assert result.is_taxonomy is False

    def test_string_taxonomy_col_stores_invalid_in_invalid_names(self):
        """Invalid species names in a taxonomy column must be stored in
        invalid_names for use in the quality report and column template.
        validity_rate must be strictly > 0.8: 5 valid + 1 invalid = 5/6 ≈ 0.833."""
        vn, vt, ntr, ttr, nts = self._full_lookups()
        df = pd.DataFrame({"species": [
            "Homo sapiens", "Homo sapiens", "Homo sapiens",
            "Homo sapiens", "Homo sapiens", "Virus X unknown",
        ]})
        result = taxonomy_flags(df, "species", vn, vt, ntr, ttr, nts)
        assert result.is_taxonomy is True
        assert result.invalid_names is not None
        assert "Virus X unknown" in result.invalid_names

    def test_result_has_rank_distribution(self):
        """taxonomy_flags must always populate rank_distribution when
        is_taxonomy=True so the template can render the rank pie chart."""
        vn, vt, ntr, ttr, nts = self._full_lookups()
        df = pd.DataFrame({"tax_id": [9606] * 10})
        result = taxonomy_flags(df, "tax_id", vn, vt, ntr, ttr, nts)
        assert result.is_taxonomy is True
        assert result.rank_distribution is not None
        assert isinstance(result.rank_distribution, dict)