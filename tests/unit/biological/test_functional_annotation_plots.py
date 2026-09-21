import pandas as pd
import pytest
import sys
import os
from unittest.mock import patch

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from biological.functional_annotation import (
    annotation_flags,
    build_annotation_lookup,
    build_cog_counts,
    cog_category_barchart,
    cog_group_donut,
    build_go_counts,
    _go_namespace_colors,
    go_term_barchart,
    go_namespace_donut,
    clean_strings,
)

COG_PATCH = "biological.functional_annotation.get_clusters_of_orthologous_groups"
GO_PATCH = "biological.functional_annotation.get_gene_ontology"


# ---------------------------------------------------------------------------
# Helpers: full-schema reference DataFrames, matching production shape
# (see data_utils/remote_data.py: COG has COG_ID/Functional Category/COG name,
# GO has GO_ID/Name/Namespace)
# ---------------------------------------------------------------------------

def full_cog_df():
    return pd.DataFrame({
        "COG_ID": ["COG0001", "COG0002", "COG0003", "COG0004"],
        "Functional Category": ["J", "K", "JK", "C"],
        "COG name": ["Ribosomal protein", "Transcription factor", "Dual function", "Energy enzyme"],
    })


def full_go_df():
    return pd.DataFrame({
        "GO_ID": ["GO:0008150", "GO:0003674", "GO:0005575", "GO:0006412"],
        "Name": ["biological_process", "molecular_function", "cellular_component", "translation"],
        "Namespace": ["BP", "MF", "CC", "BP"],
    })


# ---------------------------------------------------------------------------
# build_annotation_lookup
# ---------------------------------------------------------------------------

class TestBuildAnnotationLookup:
    def test_returns_cleaned_and_raw_keys(self):
        df = pd.DataFrame({"COG_ID": ["cog0001", " COG0002 "]})
        result = build_annotation_lookup(df, "COG_ID")
        assert set(result.keys()) == {"cleaned", "raw"}

    def test_cleaned_is_uppercased_and_stripped(self):
        df = pd.DataFrame({"COG_ID": ["cog0001", " COG0002 "]})
        result = build_annotation_lookup(df, "COG_ID")
        assert result["cleaned"] == {"COG0001", "COG0002"}

    def test_raw_preserves_original_casing(self):
        df = pd.DataFrame({"COG_ID": ["cog0001", "COG0002"]})
        result = build_annotation_lookup(df, "COG_ID")
        assert "cog0001" in result["raw"]
        assert "COG0002" in result["raw"]


# ---------------------------------------------------------------------------
# build_cog_counts
# ---------------------------------------------------------------------------

class TestBuildCogCounts:
    def test_matches_and_counts_categories(self):
        cog_df = full_cog_df()
        col = pd.Series(["COG0001"] * 5 + ["COG0002"] * 3 + ["COG0003"] * 2)
        counts, invalid = build_cog_counts(col, cog_df)
        assert invalid is None
        # COG0003 has category "JK" -> exploded into J and K rows too
        categories = set(counts["category"])
        assert "J" in categories
        assert "K" in categories
        assert "count" in counts.columns

    def test_invalid_ids_reported(self):
        cog_df = full_cog_df()
        col = pd.Series(["COG0001"] * 5 + ["UNKNOWN1", "UNKNOWN2"])
        counts, invalid = build_cog_counts(col, cog_df)
        assert invalid == ["UNKNOWN1", "UNKNOWN2"]

    def test_all_invalid_returns_empty_counts(self):
        """Documents current behavior: when nothing matches at all, the
        empty-match branch returns None for `invalid` unconditionally
        (the computed invalid_values list is discarded in that branch) —
        unlike build_go_counts, which does propagate invalid IDs in the
        equivalent all-invalid case. Update this test if that asymmetry
        is ever fixed to also surface invalid_values here."""
        cog_df = full_cog_df()
        col = pd.Series(["NOPE1", "NOPE2"])
        counts, invalid = build_cog_counts(col, cog_df)
        assert counts.empty
        assert invalid is None

    def test_no_invalid_returns_none(self):
        cog_df = full_cog_df()
        col = pd.Series(["COG0001", "COG0002"])
        _, invalid = build_cog_counts(col, cog_df)
        assert invalid is None

    def test_meta_columns_attached(self):
        cog_df = full_cog_df()
        col = pd.Series(["COG0001"] * 3)
        counts, _ = build_cog_counts(col, cog_df)
        assert list(counts.columns) == ["category", "group", "color", "description", "count"]
        row = counts[counts["category"] == "J"].iloc[0]
        assert row["group"] == "INFORMATION STORAGE AND PROCESSING"

    def test_custom_matched_column(self):
        cog_df = full_cog_df()
        col = pd.Series(["Ribosomal protein"] * 4)
        counts, invalid = build_cog_counts(col, cog_df, cog_col="COG name")
        assert invalid is None
        assert "J" in set(counts["category"])


# ---------------------------------------------------------------------------
# cog_category_barchart / cog_group_donut
# ---------------------------------------------------------------------------

class TestCogCharts:
    def test_barchart_none_for_empty_counts(self):
        empty = pd.DataFrame(columns=["category", "group", "color", "description", "count"])
        assert cog_category_barchart(empty) is None

    def test_barchart_returns_html_for_nonempty_counts(self):
        cog_df = full_cog_df()
        col = pd.Series(["COG0001"] * 5 + ["COG0002"] * 3)
        counts, _ = build_cog_counts(col, cog_df)
        html = cog_category_barchart(counts)
        assert isinstance(html, str)
        assert len(html) > 0

    def test_group_donut_none_for_empty_counts(self):
        empty = pd.DataFrame(columns=["category", "group", "color", "description", "count"])
        assert cog_group_donut(empty) is None

    def test_group_donut_returns_html_for_nonempty_counts(self):
        cog_df = full_cog_df()
        col = pd.Series(["COG0001"] * 5 + ["COG0004"] * 2)
        counts, _ = build_cog_counts(col, cog_df)
        html = cog_group_donut(counts)
        assert isinstance(html, str)


# ---------------------------------------------------------------------------
# build_go_counts
# ---------------------------------------------------------------------------

class TestBuildGoCounts:
    def test_matches_and_counts_terms(self):
        go_df = full_go_df()
        col = pd.Series(["GO:0008150"] * 5 + ["GO:0006412"] * 3)
        term_counts, namespace_counts, invalid = build_go_counts(col, go_df)
        assert invalid is None
        assert set(term_counts["GO_ID"]) == {"GO:0008150", "GO:0006412"}
        assert "Namespace" in namespace_counts.columns
        assert "count" in namespace_counts.columns

    def test_invalid_ids_reported(self):
        go_df = full_go_df()
        col = pd.Series(["GO:0008150"] * 5 + ["GO:9999999"])
        _, _, invalid = build_go_counts(col, go_df)
        assert invalid == ["GO:9999999"]

    def test_all_invalid_returns_empty_frames(self):
        go_df = full_go_df()
        col = pd.Series(["GO:BAD1", "GO:BAD2"])
        term_counts, namespace_counts, invalid = build_go_counts(col, go_df)
        assert term_counts.empty
        assert namespace_counts.empty
        assert invalid == ["GO:BAD1", "GO:BAD2"]

    def test_top_n_limits_returned_terms(self):
        go_df = pd.DataFrame({
            "GO_ID": [f"GO:{i:07d}" for i in range(20)],
            "Name": [f"term{i}" for i in range(20)],
            "Namespace": ["BP"] * 20,
        })
        col = pd.Series([f"GO:{i:07d}" for i in range(20) for _ in range(i + 1)])
        term_counts, _, _ = build_go_counts(col, go_df, top_n=5)
        assert len(term_counts) <= 5

    def test_namespace_with_null_rows_dropped_from_reference(self):
        go_df = full_go_df()
        go_df.loc[len(go_df)] = [None, "orphan", "BP"]
        col = pd.Series(["GO:0008150"] * 3)
        term_counts, _, _ = build_go_counts(col, go_df)
        assert not term_counts.empty


# ---------------------------------------------------------------------------
# _go_namespace_colors
# ---------------------------------------------------------------------------

class TestGoNamespaceColors:
    def test_assigns_a_color_per_namespace(self):
        colors = _go_namespace_colors(["BP", "MF", "CC"])
        assert set(colors.keys()) == {"BP", "MF", "CC"}

    def test_colors_are_deterministic_for_same_input(self):
        colors1 = _go_namespace_colors(["BP", "MF"])
        colors2 = _go_namespace_colors(["MF", "BP"])
        assert colors1 == colors2

    def test_cycles_palette_when_more_namespaces_than_colors(self):
        namespaces = ["a", "b", "c", "d", "e"]
        colors = _go_namespace_colors(namespaces)
        # palette has 3 colors; 4th namespace should repeat the 1st color
        sorted_ns = sorted(namespaces)
        assert colors[sorted_ns[0]] == colors[sorted_ns[3]]


# ---------------------------------------------------------------------------
# go_term_barchart / go_namespace_donut
# ---------------------------------------------------------------------------

class TestGoCharts:
    def test_term_barchart_none_for_empty(self):
        go_df = full_go_df()
        empty = pd.DataFrame(columns=list(go_df.columns) + ["count"])
        assert go_term_barchart(empty) is None

    def test_term_barchart_returns_html(self):
        go_df = full_go_df()
        col = pd.Series(["GO:0008150"] * 5 + ["GO:0006412"] * 2)
        term_counts, _, _ = build_go_counts(col, go_df)
        html = go_term_barchart(term_counts)
        assert isinstance(html, str)

    def test_namespace_donut_none_for_empty(self):
        empty = pd.DataFrame(columns=["Namespace", "count"])
        assert go_namespace_donut(empty) is None

    def test_namespace_donut_returns_html(self):
        go_df = full_go_df()
        col = pd.Series(["GO:0008150"] * 5 + ["GO:0006412"] * 2)
        _, namespace_counts, _ = build_go_counts(col, go_df)
        html = go_namespace_donut(namespace_counts)
        assert isinstance(html, str)


# ---------------------------------------------------------------------------
# annotation_flags end-to-end with full-schema reference data (realistic
# production shape) — exercises the whole cog/go plot-building pipeline
# through the public entry point, not just the unit-level helpers above.
# ---------------------------------------------------------------------------

class TestAnnotationFlagsFullSchema:
    def test_cog_end_to_end_produces_charts(self):
        df = pd.DataFrame({"cog": ["COG0001"] * 6 + ["COG0002"] * 4})
        with patch(COG_PATCH, return_value=full_cog_df()):
            result = annotation_flags(df, "cog", "cog")
        assert result.is_annotation is True
        assert result.cog_bar is not None
        assert result.cog_group is not None
        assert result.cog_invalid is None

    def test_cog_end_to_end_with_invalid_values(self):
        df = pd.DataFrame({"cog": ["COG0001"] * 9 + ["UNKNOWN"]})
        with patch(COG_PATCH, return_value=full_cog_df()):
            result = annotation_flags(df, "cog", "cog")
        assert result.cog_invalid == ["UNKNOWN"]
        assert result.cog_bar is not None

    def test_go_end_to_end_produces_charts(self):
        df = pd.DataFrame({"go": ["GO:0008150"] * 6 + ["GO:0006412"] * 4})
        with patch(GO_PATCH, return_value=full_go_df()):
            result = annotation_flags(df, "go", "go")
        assert result.is_annotation is True
        assert result.go_bar is not None
        assert result.go_name_count is not None
        assert result.go_invalid is None

    def test_go_end_to_end_with_invalid_values(self):
        df = pd.DataFrame({"go": ["GO:0008150"] * 9 + ["GO:BAD"]})
        with patch(GO_PATCH, return_value=full_go_df()):
            result = annotation_flags(df, "go", "go")
        assert result.go_invalid == ["GO:BAD"]
        assert result.go_bar is not None

    def test_cog_matched_by_name_when_id_invalid(self):
        """COG_ID column doesn't match, but 'COG name' does — short-circuit
        must still find the name-based match without crashing."""
        df = pd.DataFrame({"cog": ["Ribosomal protein"] * 10})
        with patch(COG_PATCH, return_value=full_cog_df()):
            result = annotation_flags(df, "cog", "cog")
        assert result.is_annotation is True
        assert result.cog_bar is not None

    def test_go_matched_by_name_when_id_invalid(self):
        df = pd.DataFrame({"go": ["translation"] * 10})
        with patch(GO_PATCH, return_value=full_go_df()):
            result = annotation_flags(df, "go", "go")
        assert result.is_annotation is True
        assert result.go_bar is not None

    def test_minimal_schema_still_works_without_plots(self):
        """Regression guard for the original bug: a reference DataFrame with
        only the ID column (no 'COG name'/'Functional Category') must not
        raise KeyError, and simply skips chart generation."""
        df = pd.DataFrame({"cog": ["J"] * 9 + ["X"]})
        minimal_cog_df = pd.DataFrame({"COG_ID": ["J", "K", "L"]})
        with patch(COG_PATCH, return_value=minimal_cog_df):
            result = annotation_flags(df, "cog", "cog")
        assert result.is_annotation is True
        assert result.cog_bar is None
        assert result.cog_group is None
        assert result.cog_invalid is None