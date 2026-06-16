import numpy as np
import pandas as pd
import pytest
import sys
import os

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from analysis.data_quality import (
    _split_numeric_string,
    check_mixed_types,
    check_suspect_values,
)


# ---------------------------------------------------------------------------
# _split_numeric_string
# ---------------------------------------------------------------------------

class TestSplitNumericString:
    def test_pure_numeric_strings_returns_none_none(self):
        """Object dtype but all values are numeric strings → no split."""
        series = pd.Series(["1.0", "2.0", "3.0"])
        numeric, string = _split_numeric_string(series)
        # All convert to numeric → string_part is empty, not None
        assert numeric is not None
        assert len(string) == 0

    def test_pure_strings_no_numeric(self):
        series = pd.Series(["alpha", "beta", "gamma"])
        numeric, string = _split_numeric_string(series)
        assert numeric is not None
        assert len(numeric) == 0
        assert len(string) == 3

    def test_mixed_numeric_and_string(self):
        series = pd.Series(["1.0", "hello", "2.5", "world"])
        numeric, string = _split_numeric_string(series)
        assert numeric is not None
        assert len(numeric) == 2
        assert len(string) == 2

    def test_mixed_inferred_dtype_path(self):
        """Series with actual int/float Python objects triggers 'mixed' inferred dtype."""
        series = pd.Series([1, "hello", 2.5, "world"])
        numeric, string = _split_numeric_string(series)
        assert numeric is not None
        assert string is not None
        assert len(numeric) == 2
        assert len(string) == 2

    def test_non_object_dtype_returns_none_none(self):
        """Pure numeric dtype (float64) has no strings to split."""
        series = pd.Series([1.0, 2.0, 3.0])  # dtype=float64
        numeric, string = _split_numeric_string(series)
        assert numeric is None
        assert string is None

    def test_integer_dtype_returns_none_none(self):
        series = pd.Series([1, 2, 3])  # dtype=int64
        numeric, string = _split_numeric_string(series)
        assert numeric is None
        assert string is None

    def test_empty_series_object_dtype(self):
        series = pd.Series([], dtype=object)
        numeric, string = _split_numeric_string(series)
        assert numeric is not None
        assert len(numeric) == 0
        assert len(string) == 0


# ---------------------------------------------------------------------------
# check_mixed_types
# ---------------------------------------------------------------------------

class TestCheckMixedTypes:
    def test_returns_none_for_all_nan_column(self):
        df = pd.DataFrame({"col": [None, None, None]})
        assert check_mixed_types(df, "col") is None

    def test_returns_none_for_pure_numeric(self):
        df = pd.DataFrame({"col": [1.0, 2.0, 3.0]})
        assert check_mixed_types(df, "col") is None

    def test_returns_none_for_pure_strings(self):
        df = pd.DataFrame({"col": ["alpha", "beta", "gamma"]})
        assert check_mixed_types(df, "col") is None

    def test_majority_numeric_minority_strings(self):
        """90 numeric strings, 2 text strings → majority Numeric, minority returned."""
        values = ["1.0"] * 90 + ["hello", "world"]
        df = pd.DataFrame({"col": values})
        result = check_mixed_types(df, "col")
        assert result is not None
        majority_type, minority = result
        assert majority_type == "Numeric"
        assert minority is not None
        assert len(minority) == 2

    def test_majority_string_minority_numerics(self):
        """90 text strings, 2 numeric strings → majority String, minority returned."""
        values = ["alpha"] * 90 + ["1.0", "2.5"]
        df = pd.DataFrame({"col": values})
        result = check_mixed_types(df, "col")
        assert result is not None
        majority_type, minority = result
        assert majority_type == "String"
        assert minority is not None

    def test_minority_none_when_ratio_above_threshold(self):
        """If minority ratio >= 0.125, minority values are not flagged."""
        # 80 numeric, 20 string → ratio 20/100 = 0.2 >= 0.125
        values = ["1.0"] * 80 + ["hello"] * 20
        df = pd.DataFrame({"col": values})
        result = check_mixed_types(df, "col")
        assert result is not None
        majority_type, minority = result
        assert majority_type == "Numeric"
        assert minority is None  # ratio too high to be suspect

    def test_minority_returned_when_ratio_below_threshold(self):
        """Minority ratio < 0.125 → minority values are returned."""
        # 100 numeric, 5 string → ratio 5/105 ≈ 0.048 < 0.125
        values = ["1.0"] * 100 + ["hello"] * 5
        df = pd.DataFrame({"col": values})
        result = check_mixed_types(df, "col")
        assert result is not None
        _, minority = result
        assert minority is not None

    def test_exact_threshold_boundary(self):
        """At exactly 0.125 ratio, minority should be None (not flagged)."""
        # 7 minority out of 56 total = 0.125 exactly
        values = ["1.0"] * 49 + ["hello"] * 7
        df = pd.DataFrame({"col": values})
        result = check_mixed_types(df, "col")
        assert result is not None
        _, minority = result
        assert minority is None

    def test_custom_threshold(self):
        """Custom suspect_threshold changes flagging behaviour."""
        values = ["1.0"] * 80 + ["hello"] * 20
        df = pd.DataFrame({"col": values})
        # With threshold=0.25, ratio 0.2 < 0.25 → minority flagged
        result = check_mixed_types(df, "col", suspect_threshold=0.25)
        assert result is not None
        _, minority = result
        assert minority is not None

    def test_returns_none_for_equal_split(self):
        """50/50 split: both parts non-empty but minority_count == majority_count.
        Neither part has len==0, so result is returned with majority determined
        by >= comparison (numeric wins ties)."""
        values = ["1.0"] * 10 + ["hello"] * 10
        df = pd.DataFrame({"col": values})
        result = check_mixed_types(df, "col")
        assert result is not None
        majority_type, _ = result
        assert majority_type == "Numeric"  # numeric >= string → Numeric wins

    def test_nan_excluded_before_split(self):
        """NaN values are dropped before type analysis."""
        values = ["1.0"] * 50 + ["hello"] * 3 + [None] * 20
        df = pd.DataFrame({"col": values})
        result = check_mixed_types(df, "col")
        assert result is not None
        majority_type, _ = result
        assert majority_type == "Numeric"

    def test_bioinformatics_mixed_column(self):
        """Realistic case: organism column with a few accidentally numeric entries."""
        values = (
            ["Escherichia coli"] * 60 +
            ["Homo sapiens"] * 30 +
            ["9606", "511145"]  # NCBI taxon IDs entered as strings
        )
        df = pd.DataFrame({"organism": values})
        result = check_mixed_types(df, "organism")
        assert result is not None
        majority_type, minority = result
        assert majority_type == "String"
        assert minority is not None
        assert len(minority) == 2


# ---------------------------------------------------------------------------
# check_suspect_values
# ---------------------------------------------------------------------------

class TestCheckSuspectValues:
    def test_returns_none_for_all_nan(self):
        df = pd.DataFrame({"col": [None, None]})
        assert check_suspect_values(df, "col") is None

    def test_returns_none_for_clean_numeric(self):
        df = pd.DataFrame({"col": [1.0, 2.0, 3.0, 4.0, 5.0]})
        assert check_suspect_values(df, "col") is None

    def test_returns_none_for_clean_strings(self):
        df = pd.DataFrame({"col": ["alpha", "beta", "gamma", "delta"]})
        assert check_suspect_values(df, "col") is None

    def test_detects_inf_in_numeric_column(self):
        df = pd.DataFrame({"col": [1.0, 2.0, np.inf, 4.0, 5.0]})
        result = check_suspect_values(df, "col")
        assert result is not None
        assert "inf" in result.values

    def test_detects_negative_inf_in_numeric_column(self):
        df = pd.DataFrame({"col": [1.0, -np.inf, 3.0, 4.0, 5.0]})
        result = check_suspect_values(df, "col")
        assert result is not None
        assert "-inf" in result.values

    def test_detects_both_inf_values(self):
        df = pd.DataFrame({"col": [1.0, np.inf, -np.inf, 4.0]})
        result = check_suspect_values(df, "col")
        assert result is not None
        assert len(result) == 2
        assert "inf" in result.values
        assert "-inf" in result.values

    def test_inf_result_is_string_series(self):
        """Infinity values are returned as string representations.
        Pandas 3.x uses StringDtype instead of object for string Series."""
        df = pd.DataFrame({"col": [1.0, np.inf, 3.0]})
        result = check_suspect_values(df, "col")
        assert result is not None
        assert result.apply(lambda x: isinstance(x, str)).all()

    def test_detects_numeric_suspects_in_object_column(self):
        """A few numeric strings in a mostly-string object column are suspect."""
        values = ["alpha"] * 40 + ["1.0", "2.5", "3.0"]
        df = pd.DataFrame({"col": values})
        result = check_suspect_values(df, "col")
        # 3/43 ≈ 0.07 < 0.15 → numeric strings flagged
        assert result is not None
        assert len(result) == 3

    def test_no_suspect_when_numeric_ratio_above_threshold(self):
        """If ≥ 15 % of object column values are numeric strings, not flagged."""
        values = ["alpha"] * 20 + ["1.0"] * 5   # 5/25 = 0.2 >= 0.15
        df = pd.DataFrame({"col": values})
        result = check_suspect_values(df, "col")
        assert result is None

    def test_exact_threshold_boundary_suspect(self):
        """Just below 0.15: flagged."""
        # 2 numeric out of 14 total = 0.1428 < 0.15 → flagged
        values = ["alpha"] * 12 + ["1.0", "2.0"]
        df = pd.DataFrame({"col": values})
        result = check_suspect_values(df, "col")
        assert result is not None

    def test_exact_threshold_boundary_not_suspect(self):
        """Exactly at 0.15: not flagged (condition is strict <)."""
        # 3 numeric out of 20 total = 0.15 → not flagged
        values = ["alpha"] * 17 + ["1.0", "2.0", "3.0"]
        df = pd.DataFrame({"col": values})
        result = check_suspect_values(df, "col")
        assert result is None

    def test_returns_none_for_pure_numeric_strings_in_object_col(self):
        """All numeric-convertible strings: no string part → no suspect."""
        df = pd.DataFrame({"col": ["1.0", "2.0", "3.0", "4.0"]})
        result = check_suspect_values(df, "col")
        # 4/4 = 1.0 >= 0.15 → not flagged
        assert result is None

    def test_nan_excluded_before_ratio_calculation(self):
        """NaN rows are dropped; ratio is computed against non-null values only."""
        values = ["alpha"] * 38 + ["1.0", "2.0"] + [None] * 20
        df = pd.DataFrame({"col": values})
        result = check_suspect_values(df, "col")
        # 2/40 = 0.05 < 0.15 → flagged
        assert result is not None

    def test_bioinformatics_quality_score_with_inf(self):
        """Quality scores should never be infinite — flags corrupt entries."""
        scores = [30.0, 35.5, 28.0, np.inf, 32.1, -np.inf]
        df = pd.DataFrame({"quality_score": scores})
        result = check_suspect_values(df, "quality_score")
        assert result is not None
        assert len(result) == 2

    def test_go_term_column_with_stray_numeric(self):
        """GO term column with a few accidental numeric entries."""
        values = (
            ["GO:0008150"] * 35 +
            ["GO:0003674"] * 25 +
            ["1234", "5678"]   # malformed entries
        )
        df = pd.DataFrame({"go_term": values})
        result = check_suspect_values(df, "go_term")
        # 2/62 ≈ 0.032 < 0.15 → flagged
        assert result is not None
        assert len(result) == 2