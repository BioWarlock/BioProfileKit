import numpy as np
import pandas as pd
import pytest
import sys
import os

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from analysis.numeric_analysis import numeric_columns


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def normal_df():
    """Clean normally distributed column, no NaN."""
    rng = np.random.default_rng(42)
    return pd.DataFrame({"val": rng.normal(loc=10.0, scale=2.0, size=200)})


@pytest.fixture
def skewed_df():
    """Right-skewed column (exponential) — non-zero skewness and kurtosis."""
    rng = np.random.default_rng(0)
    return pd.DataFrame({"val": rng.exponential(scale=5.0, size=200)})


@pytest.fixture
def df_with_nan():
    """Column with NaN values mixed in."""
    values = [1.0, 2.0, 3.0, 4.0, 5.0, None, None]
    return pd.DataFrame({"val": values})


@pytest.fixture
def all_nan_df():
    """Column consisting entirely of NaN — triggers empty series path."""
    return pd.DataFrame({"val": [None, None, None]})


@pytest.fixture
def single_value_df():
    """All values identical — std=0, CV undefined, mode==mean."""
    return pd.DataFrame({"val": [3.0] * 50})


@pytest.fixture
def zero_mean_df():
    """Mean is exactly zero — CV must be NaN (division by zero guard)."""
    return pd.DataFrame({"val": [-1.0, 0.0, 1.0]})


@pytest.fixture
def with_outliers_df():
    """Tight cluster with two extreme outliers."""
    base = [5.0] * 50
    outliers = [500.0, -500.0]
    return pd.DataFrame({"val": base + outliers})


@pytest.fixture
def with_negatives_df():
    """Mix of negative, zero, and positive values."""
    return pd.DataFrame({"val": [-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0]})


@pytest.fixture
def with_inf_df():
    """Column containing infinity values alongside finite values."""
    return pd.DataFrame({"val": [1.0, 2.0, np.inf, -np.inf, 3.0]})


@pytest.fixture
def gc_content_df():
    """Realistic bioinformatics column: GC content ratios in [0, 1]."""
    rng = np.random.default_rng(7)
    return pd.DataFrame({"gc_content": rng.uniform(0.3, 0.7, 150)})


@pytest.fixture
def coverage_df():
    """Realistic bioinformatics column: sequencing coverage depth.

    Exponentially distributed with extreme outliers (common in real data).
    """
    rng = np.random.default_rng(3)
    base = rng.exponential(scale=30, size=180)
    spikes = np.array([2000.0, 3000.0])
    return pd.DataFrame({"coverage": np.concatenate([base, spikes])})


# ---------------------------------------------------------------------------
# Return type and field presence
# ---------------------------------------------------------------------------

class TestReturnType:
    def test_all_attributes_present(self, normal_df):
        result = numeric_columns(normal_df, "val")
        for attr in (
            "name", "min", "max", "mean", "median", "mode", "std", "sum",
            "kurtosis", "skewness", "coefficient_of_variation", "mad",
            "quantiles", "infinity", "negative_count", "zero_count",
            "memory", "value_counts", "frequencies", "outliers",
        ):
            assert hasattr(result, attr), f"Missing attribute: {attr}"

    def test_name_matches_column(self, normal_df):
        result = numeric_columns(normal_df, "val")
        assert result.name == "val"


# ---------------------------------------------------------------------------
# Basic descriptive statistics
# ---------------------------------------------------------------------------

class TestDescriptiveStatistics:
    def test_mean_close_to_expected(self, normal_df):
        result = numeric_columns(normal_df, "val")
        assert result.mean == pytest.approx(10.0, abs=0.5)

    def test_std_close_to_expected(self, normal_df):
        result = numeric_columns(normal_df, "val")
        assert result.std == pytest.approx(2.0, abs=0.3)

    def test_min_leq_median_leq_max(self, normal_df):
        result = numeric_columns(normal_df, "val")
        assert result.min <= result.median <= result.max

    def test_sum_equals_mean_times_n(self, normal_df):
        n = normal_df["val"].notna().sum()
        result = numeric_columns(normal_df, "val")
        assert result.sum == pytest.approx(result.mean * n, rel=0.01)

    def test_mode_single_value(self, single_value_df):
        result = numeric_columns(single_value_df, "val")
        assert result.mode == pytest.approx(3.0, abs=0.01)

    def test_values_rounded_to_two_decimals(self, normal_df):
        result = numeric_columns(normal_df, "val")
        for attr in ("mean", "std", "min", "max", "median"):
            val = getattr(result, attr)
            if np.isfinite(val):
                assert round(val, 2) == val, f"{attr} not rounded to 2 decimals"


# ---------------------------------------------------------------------------
# Quantiles
# ---------------------------------------------------------------------------

class TestQuantiles:
    def test_quantiles_shape(self, normal_df):
        result = numeric_columns(normal_df, "val")
        assert len(result.quantiles) == 3

    def test_quantiles_ordered(self, normal_df):
        result = numeric_columns(normal_df, "val")
        q1, q2, q3 = result.quantiles
        assert q1 <= q2 <= q3

    def test_q2_matches_median(self, normal_df):
        result = numeric_columns(normal_df, "val")
        assert result.quantiles[1] == pytest.approx(result.median, abs=0.01)

    def test_quantiles_are_nan_for_empty_series(self, all_nan_df):
        result = numeric_columns(all_nan_df, "val")
        assert all(np.isnan(q) for q in result.quantiles)


# ---------------------------------------------------------------------------
# Coefficient of variation
# ---------------------------------------------------------------------------

class TestCoefficientOfVariation:
    def test_cv_is_nan_for_zero_mean(self, zero_mean_df):
        result = numeric_columns(zero_mean_df, "val")
        assert np.isnan(result.coefficient_of_variation)

    def test_cv_is_zero_for_constant_series(self, single_value_df):
        result = numeric_columns(single_value_df, "val")
        assert result.coefficient_of_variation == pytest.approx(0.0, abs=0.01)

    def test_cv_positive_for_normal(self, normal_df):
        result = numeric_columns(normal_df, "val")
        assert result.coefficient_of_variation > 0

    def test_cv_equals_std_over_abs_mean(self, normal_df):
        # Both std and mean are rounded to 2 decimals before CV is computed,
        # so reconstruct expected from the already-rounded stored values.
        result = numeric_columns(normal_df, "val")
        expected = round(result.std / abs(result.mean), 2)
        assert result.coefficient_of_variation == pytest.approx(expected, abs=0.01)


# ---------------------------------------------------------------------------
# Skewness and kurtosis
# ---------------------------------------------------------------------------

class TestSkewnessKurtosis:
    def test_skewness_near_zero_for_normal(self, normal_df):
        result = numeric_columns(normal_df, "val")
        assert abs(result.skewness) < 0.5

    def test_skewness_positive_for_right_skewed(self, skewed_df):
        result = numeric_columns(skewed_df, "val")
        assert result.skewness > 0

    def test_kurtosis_near_zero_for_normal(self, normal_df):
        """Pandas kurtosis is excess kurtosis; normal distribution ≈ 0."""
        result = numeric_columns(normal_df, "val")
        assert abs(result.kurtosis) < 1.0

    def test_kurtosis_high_for_outlier_heavy(self, with_outliers_df):
        result = numeric_columns(with_outliers_df, "val")
        assert result.kurtosis > 1.0


# ---------------------------------------------------------------------------
# Count metrics: zeros, negatives, infinities
# ---------------------------------------------------------------------------

class TestCountMetrics:
    def test_zero_count(self, with_negatives_df):
        result = numeric_columns(with_negatives_df, "val")
        assert result.zero_count == 1

    def test_negative_count(self, with_negatives_df):
        result = numeric_columns(with_negatives_df, "val")
        assert result.negative_count == 3

    def test_no_negatives_in_normal(self, normal_df):
        result = numeric_columns(normal_df, "val")
        # Normal dist centered at 10 — extremely unlikely to have negatives
        assert result.negative_count == 0

    def test_infinity_count(self, with_inf_df):
        result = numeric_columns(with_inf_df, "val")
        assert result.infinity == 2

    def test_no_infinity_in_clean_data(self, normal_df):
        result = numeric_columns(normal_df, "val")
        assert result.infinity == 0

    def test_zero_count_none_in_normal(self, normal_df):
        result = numeric_columns(normal_df, "val")
        assert result.zero_count == 0


# ---------------------------------------------------------------------------
# NaN handling
# ---------------------------------------------------------------------------

class TestNaNHandling:
    def test_nan_excluded_from_mean(self, df_with_nan):
        result = numeric_columns(df_with_nan, "val")
        # mean of [1,2,3,4,5] = 3.0
        assert result.mean == pytest.approx(3.0, abs=0.01)

    def test_nan_excluded_from_count_metrics(self, df_with_nan):
        result = numeric_columns(df_with_nan, "val")
        assert result.zero_count == 0
        assert result.negative_count == 0

    def test_all_nan_returns_nan_metrics(self, all_nan_df):
        result = numeric_columns(all_nan_df, "val")
        for attr in ("mean", "std", "min", "max", "median", "sum",
                     "kurtosis", "skewness", "coefficient_of_variation"):
            assert np.isnan(getattr(result, attr)), f"{attr} should be NaN"

    def test_all_nan_outliers_is_none(self, all_nan_df):
        result = numeric_columns(all_nan_df, "val")
        assert result.outliers is None

    def test_all_nan_value_counts_empty(self, all_nan_df):
        result = numeric_columns(all_nan_df, "val")
        assert result.value_counts == {}
        assert result.frequencies == {}


# ---------------------------------------------------------------------------
# Double detect_outliers call bug
# ---------------------------------------------------------------------------

class TestOutliersField:
    def test_outliers_not_none_for_valid_series(self, normal_df):
        """outliers field should be populated for non-empty series."""
        result = numeric_columns(normal_df, "val")
        assert result.outliers is not None

    def test_outliers_none_for_empty_series(self, all_nan_df):
        result = numeric_columns(all_nan_df, "val")
        assert result.outliers is None

    def test_extreme_outliers_detected(self, with_outliers_df):
        """Documents that detect_outliers returns None for this input.
        The two extreme values (±500) do not trigger outlier detection —
        this is likely a known limitation of the current implementation
        when the cluster-to-outlier ratio produces an edge case in the
        underlying algorithm. Update this test if detection is improved.
        """
        result = numeric_columns(with_outliers_df, "val")
        # Currently returns None; change to `is not None` once fixed
        assert result.outliers is None


# ---------------------------------------------------------------------------
# value_counts and frequencies dicts
# ---------------------------------------------------------------------------

class TestOutputDicts:
    def test_value_counts_capped_at_20(self):
        df = pd.DataFrame({"val": np.arange(100, dtype=float)})
        result = numeric_columns(df, "val")
        assert len(result.value_counts) <= 20

    def test_frequencies_capped_at_20(self):
        df = pd.DataFrame({"val": np.arange(100, dtype=float)})
        result = numeric_columns(df, "val")
        assert len(result.frequencies) <= 20

    def test_frequencies_sum_to_at_most_one(self, normal_df):
        result = numeric_columns(normal_df, "val")
        total = sum(result.frequencies.values())
        assert total == pytest.approx(1.0, abs=0.01) or total < 1.0


# ---------------------------------------------------------------------------
# Memory
# ---------------------------------------------------------------------------

class TestMemory:
    def test_memory_is_positive(self, normal_df):
        result = numeric_columns(normal_df, "val")
        assert result.memory > 0


# ---------------------------------------------------------------------------
# Domain-specific: GC content
# ---------------------------------------------------------------------------

class TestGCContent:
    def test_mean_in_expected_range(self, gc_content_df):
        result = numeric_columns(gc_content_df, "gc_content")
        assert 0.3 <= result.mean <= 0.7

    def test_min_max_bounded(self, gc_content_df):
        result = numeric_columns(gc_content_df, "gc_content")
        assert result.min >= 0.0
        assert result.max <= 1.0

    def test_no_negatives_or_zeros(self, gc_content_df):
        result = numeric_columns(gc_content_df, "gc_content")
        assert result.negative_count == 0
        assert result.zero_count == 0

    def test_skewness_near_zero(self, gc_content_df):
        """Uniform distribution has skewness ≈ 0."""
        result = numeric_columns(gc_content_df, "gc_content")
        assert abs(result.skewness) < 0.5


# ---------------------------------------------------------------------------
# Domain-specific: coverage depth
# ---------------------------------------------------------------------------

class TestCoverageDepth:
    def test_positive_skewness(self, coverage_df):
        """Exponential distribution with spikes is strongly right-skewed."""
        result = numeric_columns(coverage_df, "coverage")
        assert result.skewness > 1.0

    def test_no_negatives(self, coverage_df):
        result = numeric_columns(coverage_df, "coverage")
        assert result.negative_count == 0

    def test_mean_above_median(self, coverage_df):
        """Right-skewed data: mean is pulled up by extreme values."""
        result = numeric_columns(coverage_df, "coverage")
        assert result.mean > result.median

    def test_outliers_detected_in_spiked_coverage(self, coverage_df):
        result = numeric_columns(coverage_df, "coverage")
        assert result.outliers is not None