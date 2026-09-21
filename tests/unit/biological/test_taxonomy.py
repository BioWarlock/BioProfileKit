import pandas as pd
import pytest
import sys
import os

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from biological.taxonomy import (
    is_taxid,
    is_taxonomy,
    taxid_rank_distribution,
    rank_distribution,
    find_outdated_names,
    build_lookups,
    build_name_to_taxid,
    build_sunburst_data,
    build_sunburst_data_from_taxids,
    taxonomy_flags,
    taxonomy_sunburst_plot,
    TaxonomyFlags,
    SUNBURST_PATH,
    SUNBURST_RANKS,
)


# ---------------------------------------------------------------------------
# is_taxid
# ---------------------------------------------------------------------------

class TestIsTaxid:
    def test_all_valid_returns_marker_string(self):
        col = pd.Series([9606, 10090, 9606], name="taxid")
        result = is_taxid(col, valid_tax_ids={9606, 10090})
        assert result == "all tax IDs valid"

    def test_invalid_ids_returned_as_set(self):
        col = pd.Series([9606] * 6 + [123456] * 4, name="taxid")
        result = is_taxid(col, valid_tax_ids={9606}, threshold=0.5)
        assert result == {123456}

    def test_excluded_column_names_return_none(self):
        for name in ("length", "start", "end", "Length", "START"):
            col = pd.Series([9606, 10090, 9606, 10090], name=name)
            assert is_taxid(col, valid_tax_ids={9606, 10090}) is None

    def test_non_numeric_column_returns_none(self):
        col = pd.Series(["abc", "def", "ghi"], name="taxid")
        assert is_taxid(col, valid_tax_ids={9606}) is None

    def test_low_validity_rate_returns_none(self):
        """Numeric column, but too few values match known tax IDs."""
        col = pd.Series([9606] * 2 + [1, 2, 3, 4, 5, 6, 7, 8], name="taxid")
        assert is_taxid(col, valid_tax_ids={9606}, threshold=0.9) is None

    def test_unnamed_column_not_excluded(self):
        col = pd.Series([9606, 9606, 9606])
        result = is_taxid(col, valid_tax_ids={9606})
        assert result == "all tax IDs valid"


# ---------------------------------------------------------------------------
# is_taxonomy
# ---------------------------------------------------------------------------

VALID_NAMES = {"Escherichia coli", "Bacillus subtilis"}
NAME_TO_RANK = {"Escherichia coli": "species", "Bacillus subtilis": "species"}
NAME_TO_SCI = {"Escherichia coli": "Escherichia coli", "Bacillus subtilis": "Bacillus subtilis"}


class TestIsTaxonomy:
    def test_returns_none_below_threshold(self):
        col = pd.Series(["Escherichia coli"] * 5 + ["Unknown organism"] * 5)
        result = is_taxonomy(col, VALID_NAMES, NAME_TO_RANK, NAME_TO_SCI)
        assert result is None

    def test_returns_valid_dict_above_threshold(self):
        col = pd.Series(["Escherichia coli"] * 9 + ["Unknown organism"])
        result = is_taxonomy(col, VALID_NAMES, NAME_TO_RANK, NAME_TO_SCI)
        assert result is not None
        assert result["valid"] is True
        # value_counts(normalize=True) drops NaN ranks by default, so the
        # one unmapped "Unknown organism" entry isn't in the denominator.
        assert result["rank_distribution"] == {"species": 1.0}

    def test_invalid_names_reported(self):
        col = pd.Series(["Escherichia coli"] * 9 + ["Unknown organism"])
        result = is_taxonomy(col, VALID_NAMES, NAME_TO_RANK, NAME_TO_SCI)
        assert result["invalid_names"] == ["Unknown organism"]

    def test_falls_back_to_cleaned_names_with_strain_suffix(self):
        """'Escherichia coli (strain K12)' should match after stripping the
        parenthetical suffix, via the cleaned-name fallback path."""
        col = pd.Series(["Escherichia coli (strain K12)"] * 9 + ["Unknown"])
        result = is_taxonomy(col, VALID_NAMES, NAME_TO_RANK, NAME_TO_SCI)
        assert result is not None
        assert result["valid"] is True

    def test_outdated_names_included(self):
        col = pd.Series(["Escherichia coli"] * 10)
        name_to_sci = {"Escherichia coli": "Escherichia coli sp. nov."}
        result = is_taxonomy(col, VALID_NAMES, NAME_TO_RANK, name_to_sci)
        assert result["outdated"] == {"Escherichia coli": "Escherichia coli sp. nov."}

    def test_no_outdated_names_is_none(self):
        col = pd.Series(["Escherichia coli"] * 10)
        result = is_taxonomy(col, VALID_NAMES, NAME_TO_RANK, NAME_TO_SCI)
        assert result["outdated"] is None

    def test_no_invalid_names_is_none(self):
        col = pd.Series(["Escherichia coli"] * 10)
        result = is_taxonomy(col, VALID_NAMES, NAME_TO_RANK, NAME_TO_SCI)
        assert result["invalid_names"] is None


# ---------------------------------------------------------------------------
# taxid_rank_distribution
# ---------------------------------------------------------------------------

class TestTaxidRankDistribution:
    def test_single_rank_not_mixed(self):
        col = pd.Series([9606, 10090, 9606, 10090])
        taxid_to_rank = {9606: "species", 10090: "species"}
        distribution, is_mixed = taxid_rank_distribution(col, taxid_to_rank)
        assert distribution == {"species": 1.0}
        assert is_mixed is False

    def test_mixed_ranks_detected(self):
        col = pd.Series([9606] * 5 + [10090] * 5)
        taxid_to_rank = {9606: "species", 10090: "genus"}
        distribution, is_mixed = taxid_rank_distribution(col, taxid_to_rank)
        assert distribution == {"species": 0.5, "genus": 0.5}
        assert is_mixed is True

    def test_rare_rank_below_threshold_not_counted_as_mixed(self):
        col = pd.Series([9606] * 19 + [10090])
        taxid_to_rank = {9606: "species", 10090: "genus"}
        distribution, is_mixed = taxid_rank_distribution(col, taxid_to_rank, threshold=0.1)
        # genus frequency = 0.05 < 0.1 threshold -> only species counts
        assert is_mixed is False


# ---------------------------------------------------------------------------
# rank_distribution
# ---------------------------------------------------------------------------

class TestRankDistribution:
    def test_basic_distribution(self):
        col = pd.Series(["Escherichia coli"] * 4 + ["Bacillus subtilis"] * 4)
        distribution, is_mixed, invalid = rank_distribution(col, NAME_TO_RANK)
        assert distribution == {"species": 1.0}
        assert invalid == []

    def test_unmapped_names_are_invalid(self):
        col = pd.Series(["Escherichia coli"] * 8 + ["Unmapped Name"] * 2)
        _, _, invalid = rank_distribution(col, NAME_TO_RANK)
        assert invalid == ["Unmapped Name"]

    def test_whitespace_stripped_before_lookup(self):
        col = pd.Series([" Escherichia coli "] * 5)
        distribution, _, invalid = rank_distribution(col, NAME_TO_RANK)
        assert distribution == {"species": 1.0}
        assert invalid == []


# ---------------------------------------------------------------------------
# find_outdated_names
# ---------------------------------------------------------------------------

class TestFindOutdatedNames:
    def test_detects_renamed_taxa(self):
        col = pd.Series(["Old Name"] * 3)
        result = find_outdated_names(col, valid_names={"Old Name"}, name_to_scientific={"Old Name": "New Name"})
        assert result == {"Old Name": "New Name"}

    def test_current_names_not_flagged(self):
        col = pd.Series(["Current Name"] * 3)
        result = find_outdated_names(
            col, valid_names={"Current Name"}, name_to_scientific={"Current Name": "Current Name"}
        )
        assert result == {}

    def test_names_not_in_valid_set_are_ignored(self):
        col = pd.Series(["Unknown"] * 3)
        result = find_outdated_names(col, valid_names={"Other"}, name_to_scientific={})
        assert result == {}


# ---------------------------------------------------------------------------
# build_lookups / build_name_to_taxid
# ---------------------------------------------------------------------------

def vocab_df():
    return pd.DataFrame({
        "tax_id": [9606, 9606, 10090],
        "name_txt": ["Homo sapiens", "Human", "Mus musculus"],
        "name_class": ["scientific name", "common name", "scientific name"],
        "rank": ["species", "species", "species"],
        "scientific_name": ["Homo sapiens", "Homo sapiens", "Mus musculus"],
    })


class TestBuildLookups:
    def test_valid_names_and_tax_ids(self):
        valid_names, valid_tax_ids, *_ = build_lookups(vocab_df())
        assert valid_names == {"Homo sapiens", "Human", "Mus musculus"}
        assert valid_tax_ids == {9606, 10090}

    def test_taxid_to_rank_uses_scientific_name_rows_only(self):
        _, _, _, taxid_to_rank, _ = build_lookups(vocab_df())
        assert taxid_to_rank.to_dict() == {9606: "species", 10090: "species"}

    def test_name_to_scientific_mapping(self):
        _, _, _, _, name_to_scientific = build_lookups(vocab_df())
        assert name_to_scientific["Human"] == "Homo sapiens"


class TestBuildNameToTaxid:
    def test_maps_names_to_taxid(self):
        result = build_name_to_taxid(vocab_df())
        assert result["Homo sapiens"] == 9606
        assert result["Mus musculus"] == 10090

    def test_duplicate_names_keep_last(self):
        df = pd.DataFrame({"name_txt": ["A", "A"], "tax_id": [1, 2]})
        result = build_name_to_taxid(df)
        assert result["A"] == 2


# ---------------------------------------------------------------------------
# build_sunburst_data / build_sunburst_data_from_taxids
# ---------------------------------------------------------------------------

def lineage_table():
    columns = ["tax_id", "scientific_name", "superdomain"] + SUNBURST_RANKS
    return pd.DataFrame([
        {
            "tax_id": 9606, "scientific_name": "Homo sapiens", "superdomain": "Eukaryota",
            "domain": "Eukaryota", "kingdom": "Metazoa", "phylum": "Chordata",
            "class": "Mammalia", "order": "Primates", "family": "Hominidae",
            "genus": "Homo", "species": "Homo sapiens",
        },
    ], columns=columns)


class TestBuildSunburstData:
    def test_builds_rows_for_matched_names(self):
        col = pd.Series(["Homo sapiens"] * 5)
        name_to_taxid = {"Homo sapiens": 9606}
        result = build_sunburst_data(col, name_to_taxid, lineage_table())
        assert len(result) == 1
        assert result.iloc[0]["count"] == 5
        assert result.iloc[0]["species"] == "Homo sapiens"

    def test_unmatched_names_are_dropped(self):
        col = pd.Series(["Unknown Organism"] * 5)
        result = build_sunburst_data(col, {}, lineage_table())
        assert result.empty

    def test_empty_lineage_table_returns_empty_with_columns(self):
        col = pd.Series(["Homo sapiens"] * 5)
        empty_lineage = pd.DataFrame(columns=["tax_id", "scientific_name", "superdomain"] + SUNBURST_RANKS)
        result = build_sunburst_data(col, {"Homo sapiens": 9606}, empty_lineage)
        assert result.empty
        assert list(result.columns) == SUNBURST_PATH + ["count"]


class TestBuildSunburstDataFromTaxids:
    def test_builds_rows_for_matched_taxids(self):
        col = pd.Series([9606] * 7)
        result = build_sunburst_data_from_taxids(col, lineage_table())
        assert len(result) == 1
        assert result.iloc[0]["count"] == 7

    def test_non_numeric_values_coerced_and_dropped(self):
        col = pd.Series(["9606", "not_a_taxid", "9606"])
        result = build_sunburst_data_from_taxids(col, lineage_table())
        assert result.iloc[0]["count"] == 2

    def test_unmatched_taxid_dropped(self):
        col = pd.Series([424242] * 3)
        result = build_sunburst_data_from_taxids(col, lineage_table())
        assert result.empty


# ---------------------------------------------------------------------------
# taxonomy_sunburst_plot — smoke test (plotly rendering itself is not our
# code to verify; we only check the function wires data through and returns
# renderable HTML without raising).
# ---------------------------------------------------------------------------

class TestTaxonomySunburstPlot:
    def test_returns_html_string(self):
        html = taxonomy_sunburst_plot(lineage_table().assign(count=[5]))
        assert isinstance(html, str)
        assert len(html) > 0


# ---------------------------------------------------------------------------
# taxonomy_flags — orchestration
# ---------------------------------------------------------------------------

class FakeLineageResolver:
    """Minimal stand-in for TaxonomyLineageResolver, returning a fixed table."""
    def __init__(self, table):
        self._table = table

    def build_lineage(self, tax_ids):
        return self._table[self._table["tax_id"].isin(tax_ids)]


class TestTaxonomyFlags:
    def test_numeric_taxid_column_all_valid(self):
        df = pd.DataFrame({"taxid": [9606, 9606, 9606, 9606]})
        result = taxonomy_flags(
            df, "taxid",
            valid_names=set(), valid_tax_ids={9606},
            name_to_rank={}, taxid_to_rank={9606: "species"},
            name_to_scientific={},
        )
        assert isinstance(result, TaxonomyFlags)
        assert result.is_taxonomy is True
        assert result.taxid == "all tax IDs valid"

    def test_string_taxonomy_column_valid(self):
        df = pd.DataFrame({"organism": ["Escherichia coli"] * 10})
        result = taxonomy_flags(
            df, "organism",
            valid_names=VALID_NAMES, valid_tax_ids=set(),
            name_to_rank=NAME_TO_RANK, taxid_to_rank={},
            name_to_scientific=NAME_TO_SCI,
        )
        assert result.is_taxonomy is True
        assert result.taxonomy == "Valid"

    def test_column_that_matches_neither_returns_not_taxonomy(self):
        df = pd.DataFrame({"notes": ["random free text " + str(i) for i in range(10)]})
        result = taxonomy_flags(
            df, "notes",
            valid_names=VALID_NAMES, valid_tax_ids={9606},
            name_to_rank=NAME_TO_RANK, taxid_to_rank={9606: "species"},
            name_to_scientific=NAME_TO_SCI,
        )
        assert result.is_taxonomy is False
        assert result.taxid is None
        assert result.taxonomy is None

    def test_sunburst_plot_built_when_lineage_resolver_provided(self):
        df = pd.DataFrame({"taxid": [9606] * 10})
        resolver = FakeLineageResolver(lineage_table())
        result = taxonomy_flags(
            df, "taxid",
            valid_names=set(), valid_tax_ids={9606},
            name_to_rank={}, taxid_to_rank={9606: "species"},
            name_to_scientific={},
            lineage_resolver=resolver,
        )
        assert result.sunburst_plot is not None
        assert isinstance(result.sunburst_plot, str)

    def test_no_sunburst_plot_without_lineage_resolver(self):
        df = pd.DataFrame({"taxid": [9606] * 10})
        result = taxonomy_flags(
            df, "taxid",
            valid_names=set(), valid_tax_ids={9606},
            name_to_rank={}, taxid_to_rank={9606: "species"},
            name_to_scientific={},
        )
        assert result.sunburst_plot is None

    def test_name_based_sunburst_plot_built(self):
        df = pd.DataFrame({"organism": ["Homo sapiens"] * 10})
        resolver = FakeLineageResolver(lineage_table())
        result = taxonomy_flags(
            df, "organism",
            valid_names={"Homo sapiens"}, valid_tax_ids=set(),
            name_to_rank={"Homo sapiens": "species"}, taxid_to_rank={},
            name_to_scientific={"Homo sapiens": "Homo sapiens"},
            name_to_taxid={"Homo sapiens": 9606},
            lineage_resolver=resolver,
        )
        assert result.sunburst_plot is not None