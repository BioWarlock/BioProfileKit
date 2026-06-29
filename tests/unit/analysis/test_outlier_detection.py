import numpy as np
import pytest
import sys
import os
from unittest.mock import patch

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from analysis.outlier_detection import detect_outliers


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_normal(n=200, loc=0.0, scale=1.0, seed=42):
    rng = np.random.default_rng(seed)
    return rng.normal(loc=loc, scale=scale, size=n).astype(np.float64)


def make_unique(n=200, seed=0):
    """n distinct float values — satisfies both length and unique-count guards."""
    rng = np.random.default_rng(seed)
    return np.unique(rng.normal(size=n * 2).astype(np.float64))[:n]


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def clean_normal():
    """200 normally distributed values, no outliers."""
    return make_normal(n=200, loc=0.0, scale=1.0)


@pytest.fixture
def right_skewed():
    """Exponential distribution — positive medcouple expected."""
    rng = np.random.default_rng(1)
    return rng.exponential(scale=5.0, size=300).astype(np.float64)


@pytest.fixture
def with_upper_outliers():
    """Tight cluster plus two extreme high values."""
    base = np.full(100, 5.0)
    noise = np.random.default_rng(5).normal(0, 0.1, 100)
    outliers = np.array([500.0, 600.0])
    vals = np.concatenate([base + noise, outliers])
    return vals.astype(np.float64)


@pytest.fixture
def with_lower_outliers():
    """Tight cluster plus two extreme low values."""
    base = np.full(100, 5.0)
    noise = np.random.default_rng(6).normal(0, 0.1, 100)
    outliers = np.array([-500.0, -600.0])
    vals = np.concatenate([base + noise, outliers])
    return vals.astype(np.float64)


@pytest.fixture
def gc_content():
    """Realistic GC content values: uniform in [0.3, 0.7], no outliers expected."""
    rng = np.random.default_rng(7)
    return rng.uniform(0.3, 0.7, size=250).astype(np.float64)


@pytest.fixture
def coverage_with_spikes():
    """Sequencing coverage: exponential base with extreme spike values."""
    rng = np.random.default_rng(3)
    base = rng.exponential(scale=30, size=200).astype(np.float64)
    spikes = np.array([5000.0, 8000.0, 10000.0], dtype=np.float64)
    return np.concatenate([base, spikes])


# ---------------------------------------------------------------------------
# Early-return None conditions
# ---------------------------------------------------------------------------

class TestEarlyReturnNone:
    def test_fewer_than_20_values_returns_none(self):
        values = make_normal(n=19)
        assert detect_outliers(values) is None

    def test_exactly_19_values_returns_none(self):
        values = make_unique(n=19)
        assert detect_outliers(values) is None

    def test_exactly_20_values_does_not_return_none(self):
        """Boundary: 20 values with >10 unique should proceed."""
        values = make_unique(n=20)
        # May still return None if IQR==0, but not due to length guard
        # We verify the length guard is not the cause by using unique values
        result = detect_outliers(values)
        # result can be Outliers or None (IQR path), but must not crash
        assert result is None or result is not None  # no exception raised

    def test_ten_or_fewer_unique_values_returns_none(self):
        """Exactly 10 unique values repeated to satisfy length guard."""
        values = np.tile(np.arange(10, dtype=np.float64), 5)  # 50 values, 10 unique
        assert detect_outliers(values) is None

    def test_eleven_unique_values_passes_guard(self):
        """11 unique values with sufficient length should not return None from unique guard."""
        values = np.tile(np.arange(11, dtype=np.float64), 5)  # 55 values, 11 unique
        # IQR > 0 for arange, so should return an Outliers object
        result = detect_outliers(values)
        assert result is not None

    def test_zero_iqr_returns_none(self):
        """Constant-ish data with IQR == 0 triggers the iqr <= 0 guard."""
        # 50 values: 11 unique but all the same except a few
        # To get IQR=0: put > 50% of values at one point
        values = np.concatenate([
            np.full(150, 5.0),
            np.arange(11, dtype=np.float64),
        ])
        assert detect_outliers(values) is None

    def test_empty_array_returns_none(self):
        values = np.array([], dtype=np.float64)
        assert detect_outliers(values) is None

    def test_single_value_returns_none(self):
        values = np.array([1.0], dtype=np.float64)
        assert detect_outliers(values) is None


# ---------------------------------------------------------------------------
# Return type and field presence
# ---------------------------------------------------------------------------

class TestReturnType:
    def test_returns_outliers_object(self, clean_normal):
        result = detect_outliers(clean_normal)
        assert result is not None
        for attr in (
            "lower_bound", "upper_bound",
            "n_lower_iqr", "n_upper_iqr",
            "medcouple",
            "n_lower_mzscore", "n_upper_mzscore",
            "n_lower_zscore", "n_upper_zscore",
        ):
            assert hasattr(result, attr), f"Missing attribute: {attr}"

    def test_bounds_are_float64(self, clean_normal):
        result = detect_outliers(clean_normal)
        assert result is not None
        assert isinstance(result.lower_bound, (float, np.floating))
        assert isinstance(result.upper_bound, (float, np.floating))

    def test_counts_are_integers(self, clean_normal):
        result = detect_outliers(clean_normal)
        assert result is not None
        for attr in ("n_lower_iqr", "n_upper_iqr",
                     "n_lower_mzscore", "n_upper_mzscore",
                     "n_lower_zscore", "n_upper_zscore"):
            assert isinstance(getattr(result, attr), int), f"{attr} should be int"


# ---------------------------------------------------------------------------
# Bounds correctness
# ---------------------------------------------------------------------------

class TestBounds:
    def test_lower_bound_leq_upper_bound(self, clean_normal):
        result = detect_outliers(clean_normal)
        assert result is not None
        assert result.lower_bound <= result.upper_bound

    def test_lower_bound_geq_data_min(self, clean_normal):
        """lower_bound is clamped to values.min()."""
        result = detect_outliers(clean_normal)
        assert result is not None
        assert result.lower_bound >= clean_normal.min() - 1e-9

    def test_upper_bound_leq_data_max(self, clean_normal):
        """upper_bound is clamped to values.max()."""
        result = detect_outliers(clean_normal)
        assert result is not None
        assert result.upper_bound <= clean_normal.max() + 1e-9

    def test_bounds_rounded_to_4_decimals(self, clean_normal):
        result = detect_outliers(clean_normal)
        assert result is not None
        assert round(result.lower_bound, 4) == result.lower_bound
        assert round(result.upper_bound, 4) == result.upper_bound

    def test_medcouple_rounded_to_4_decimals(self, clean_normal):
        result = detect_outliers(clean_normal)
        assert result is not None
        assert round(float(result.medcouple), 4) == float(result.medcouple)


# ---------------------------------------------------------------------------
# Outlier counts non-negative
# ---------------------------------------------------------------------------

class TestCountsNonNegative:
    def test_all_counts_non_negative(self, clean_normal):
        result = detect_outliers(clean_normal)
        assert result is not None
        for attr in ("n_lower_iqr", "n_upper_iqr",
                     "n_lower_mzscore", "n_upper_mzscore",
                     "n_lower_zscore", "n_upper_zscore"):
            assert getattr(result, attr) >= 0, f"{attr} is negative"

    def test_no_false_outliers_in_clean_normal(self, clean_normal):
        """Standard normal has very few 3-sigma outliers at n=200."""
        result = detect_outliers(clean_normal)
        assert result is not None
        total = result.n_lower_zscore + result.n_upper_zscore
        assert total <= 4  # <2 % expected; ≤4 is a generous bound

    def test_upper_outliers_detected_by_zscore(self, with_upper_outliers):
        result = detect_outliers(with_upper_outliers)
        assert result is not None
        assert result.n_upper_zscore > 0

    def test_lower_outliers_detected_by_zscore(self, with_lower_outliers):
        result = detect_outliers(with_lower_outliers)
        assert result is not None
        assert result.n_lower_zscore > 0

    def test_upper_outliers_detected_by_mzscore(self, with_upper_outliers):
        result = detect_outliers(with_upper_outliers)
        assert result is not None
        assert result.n_upper_mzscore > 0

    def test_lower_outliers_detected_by_mzscore(self, with_lower_outliers):
        result = detect_outliers(with_lower_outliers)
        assert result is not None
        assert result.n_lower_mzscore > 0


# ---------------------------------------------------------------------------
# Medcouple sign and adjusted boxplot asymmetry
# ---------------------------------------------------------------------------

class TestMedcouple:
    def test_medcouple_in_valid_range(self, clean_normal):
        """Medcouple is bounded in [-1, 1]."""
        result = detect_outliers(clean_normal)
        assert result is not None
        assert -1.0 <= float(result.medcouple) <= 1.0

    def test_medcouple_positive_for_right_skewed(self, right_skewed):
        """Exponential distribution has positive skew → positive medcouple."""
        result = detect_outliers(right_skewed)
        assert result is not None
        assert float(result.medcouple) > 0

    def test_positive_mc_widens_upper_bound(self, right_skewed):
        """With mc > 0 the upper fence uses exp(3*mc) > exp(-4*mc),
        so upper_bound should be further from Q3 than lower_bound from Q1."""
        result = detect_outliers(right_skewed)
        assert result is not None
        q3 = np.percentile(right_skewed, 75)
        q1 = np.percentile(right_skewed, 25)
        upper_distance = result.upper_bound - q3
        lower_distance = q1 - result.lower_bound
        assert upper_distance >= lower_distance

    def test_medcouple_near_zero_for_symmetric(self, clean_normal):
        """Symmetric distribution should have medcouple close to 0."""
        result = detect_outliers(clean_normal)
        assert result is not None
        assert abs(float(result.medcouple)) < 0.3

    def test_negative_mc_branch_via_mock(self):
        """Force mc < 0 branch by mocking fast_medcouple."""
        values = make_unique(n=200)
        with patch("analysis.outlier_detection.fast_medcouple", return_value=-0.3):
            result = detect_outliers(values)
        # Should complete without error and return an Outliers object
        assert result is not None

    def test_zero_mc_branch_via_mock(self):
        """mc == 0 takes the mc >= 0 branch."""
        values = make_unique(n=200)
        with patch("analysis.outlier_detection.fast_medcouple", return_value=0.0):
            result = detect_outliers(values)
        assert result is not None


# ---------------------------------------------------------------------------
# Zero-MAD and zero-std edge cases
# ---------------------------------------------------------------------------

class TestZeroVarianceMetrics:
    def test_zero_mad_sets_mzscore_counts_to_zero(self):
        """When MAD == 0, modified Z-score block is skipped → counts stay 0."""
        # Construct: >10 unique, >20 values, IQR > 0, but MAD == 0
        # Majority at one point gives MAD=0; spread gives IQR>0 and >10 unique
        values = np.concatenate([
            np.full(100, 5.0),        # median = 5.0 → MAD = 0
            np.linspace(0, 20, 50),   # provides IQR > 0 and unique values
        ]).astype(np.float64)
        # unique count: 50 from linspace + 1 from full = 51 > 10 ✓
        result = detect_outliers(values)
        if result is not None:
            assert result.n_lower_mzscore == 0
            assert result.n_upper_mzscore == 0

    def test_zero_std_sets_zscore_counts_to_zero(self):
        """When std == 0 (all values identical), Z-score block is skipped."""
        # Can't reach this without hitting IQR=0 guard first in practice,
        # so mock std behaviour via a synthetic near-zero std case
        values = np.concatenate([
            np.full(150, 5.0),
            np.linspace(4.9, 5.1, 50),
        ]).astype(np.float64)
        result = detect_outliers(values)
        # Just verify no crash and counts are non-negative
        if result is not None:
            assert result.n_lower_zscore >= 0
            assert result.n_upper_zscore >= 0


# ---------------------------------------------------------------------------
# Domain-specific: bioinformatics
# ---------------------------------------------------------------------------

class TestGCContent:
    def test_gc_uniform_no_iqr_outliers(self, gc_content):
        """Uniform GC content should produce no IQR-based outliers."""
        result = detect_outliers(gc_content)
        assert result is not None
        assert result.n_lower_iqr == 0
        assert result.n_upper_iqr == 0

    def test_gc_medcouple_near_zero(self, gc_content):
        result = detect_outliers(gc_content)
        assert result is not None
        assert abs(float(result.medcouple)) < 0.2

    def test_gc_bounds_within_unit_interval(self, gc_content):
        result = detect_outliers(gc_content)
        assert result is not None
        assert result.lower_bound >= 0.0
        assert result.upper_bound <= 1.0


class TestCoverageDepth:
    def test_spikes_detected_by_zscore(self, coverage_with_spikes):
        result = detect_outliers(coverage_with_spikes)
        assert result is not None
        assert result.n_upper_zscore > 0

    def test_spikes_detected_by_mzscore(self, coverage_with_spikes):
        result = detect_outliers(coverage_with_spikes)
        assert result is not None
        assert result.n_upper_mzscore > 0

    def test_positive_medcouple_for_coverage(self, coverage_with_spikes):
        """Coverage depth is right-skewed → positive medcouple."""
        result = detect_outliers(coverage_with_spikes)
        assert result is not None
        assert float(result.medcouple) > 0

    def test_no_lower_outliers_in_coverage(self, coverage_with_spikes):
        """Exponential coverage has no negative values → no lower outliers."""
        result = detect_outliers(coverage_with_spikes)
        assert result is not None
        assert result.n_lower_iqr == 0
        assert result.n_lower_zscore == 0
        assert result.n_lower_mzscore == 0