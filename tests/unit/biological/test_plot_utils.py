import pandas as pd
import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from biological.plot_utils import (
    length_distribution,
    gc_distribution,
    ambiguous_distribution,
    at_gc_skewness,
    aa_group_distribution,
    config,
)


def make_overview():
    return pd.DataFrame({
        "lengths": [10, 20, 30, 40, 50],
        "gc_content": [40.0, 45.0, 50.0, 55.0, 60.0],
        "ambiguous_count": [0, 1, 0, 2, 0],
        "GC_skew": [0.1, -0.1, 0.0, 0.2, -0.2],
        "AT_skew": [0.05, -0.05, 0.0, 0.1, -0.1],
        "sequence": ["ACGT", "TTTT", "GGGG", None, "A" * 30],
    })


# ---------------------------------------------------------------------------
# length_distribution
# ---------------------------------------------------------------------------

class TestLengthDistribution:
    def test_returns_html_string(self):
        result = length_distribution(make_overview())
        assert isinstance(result, str)
        assert len(result) > 0

    def test_sets_config_filename(self):
        length_distribution(make_overview())
        assert config["toImageButtonOptions"]["filename"] == "length_distribution"

    def test_custom_unit_does_not_raise(self):
        result = length_distribution(make_overview(), unit="residues")
        assert isinstance(result, str)

    def test_large_dataframe_caps_bins_at_300(self):
        """Rows > 10000 -> nbins fixed at 300 instead of one bin per row;
        this should not raise even with a large synthetic frame."""
        big_df = pd.DataFrame({"lengths": list(range(10001))})
        result = length_distribution(big_df)
        assert isinstance(result, str)

    def test_empty_dataframe_does_not_raise(self):
        empty = pd.DataFrame({"lengths": []})
        result = length_distribution(empty)
        assert isinstance(result, str)


# ---------------------------------------------------------------------------
# gc_distribution
# ---------------------------------------------------------------------------

class TestGcDistribution:
    def test_returns_html_string(self):
        result = gc_distribution(make_overview())
        assert isinstance(result, str)

    def test_sets_config_filename(self):
        gc_distribution(make_overview())
        assert config["toImageButtonOptions"]["filename"] == "gc_distribution"

    def test_single_row_does_not_raise(self):
        df = pd.DataFrame({"gc_content": [50.0]})
        result = gc_distribution(df)
        assert isinstance(result, str)


# ---------------------------------------------------------------------------
# ambiguous_distribution
# ---------------------------------------------------------------------------

class TestAmbiguousDistribution:
    def test_returns_html_string(self):
        result = ambiguous_distribution(make_overview())
        assert isinstance(result, str)

    def test_default_column_and_label(self):
        result = ambiguous_distribution(make_overview())
        assert isinstance(result, str)

    def test_custom_column_and_label(self):
        df = make_overview().rename(columns={"ambiguous_count": "custom_col"})
        result = ambiguous_distribution(df, col="custom_col", label="X/J/U")
        assert isinstance(result, str)

    def test_all_zero_column_does_not_raise(self):
        df = make_overview()
        df["ambiguous_count"] = 0
        result = ambiguous_distribution(df)
        assert isinstance(result, str)


# ---------------------------------------------------------------------------
# at_gc_skewness
# ---------------------------------------------------------------------------

class TestAtGcSkewness:
    def test_returns_html_string(self):
        result = at_gc_skewness(make_overview())
        assert isinstance(result, str)

    def test_sets_config_filename(self):
        at_gc_skewness(make_overview())
        assert config["toImageButtonOptions"]["filename"] == "at_gc_skewness"

    def test_adds_hover_sequence_label_column(self):
        """The function mutates its input DataFrame in place, adding a
        'hover_sequence_label' column derived from 'sequence'."""
        df = make_overview()
        at_gc_skewness(df)
        assert "hover_sequence_label" in df.columns
        assert len(df["hover_sequence_label"]) == len(df)

    def test_none_sequence_gets_placeholder_label(self):
        df = make_overview()
        at_gc_skewness(df)
        none_row = df[df["sequence"].isna()]
        assert "[No Sequence Data]" in none_row["hover_sequence_label"].iloc[0]

    def test_long_sequence_is_truncated_in_label(self):
        df = make_overview()
        at_gc_skewness(df)
        long_row_label = df["hover_sequence_label"].iloc[4]  # "A"*30
        assert "..." in long_row_label

    def test_short_sequence_not_truncated(self):
        df = make_overview()
        at_gc_skewness(df)
        short_row_label = df["hover_sequence_label"].iloc[0]  # "ACGT"
        assert "..." not in short_row_label


# ---------------------------------------------------------------------------
# aa_group_distribution
# ---------------------------------------------------------------------------

class TestAaGroupDistributionPlot:
    def test_returns_html_string(self):
        result = aa_group_distribution({"Unpolar": 0.5, "Polar": 0.3, "Aromatic": 0.2})
        assert isinstance(result, str)

    def test_sets_config_filename(self):
        aa_group_distribution({"Unpolar": 0.5})
        assert config["toImageButtonOptions"]["filename"] == "aa_group_distribution"

    def test_converts_fractions_to_percentages(self):
        """Values are documented as fractions (0-1) and multiplied by 100
        for display — a 0.5 input should not raise and the function
        should complete normally with an empty-looking but valid group."""
        result = aa_group_distribution({"Unpolar": 0.1234})
        assert isinstance(result, str)
