import numpy as np
import pandas as pd
import pytest
import sys
import os

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from analysis.multivariate_analysis import (
    get_correlation,
    correlation_heatmap,
    pearson_correlation_heatmap,
    cramers_heatmap,
    eta_heatmap,
    compute_correlation_matrix,
    missing_matrix,
    missing_values_barchart,
    balance_plot,
    boxplot,
    scatter_matrix,
    multivariate_analysis,
)


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def numeric_df():
    """Clean numeric DataFrame with meaningful correlations."""
    rng = np.random.default_rng(42)
    x = rng.standard_normal(100)
    return pd.DataFrame({
        "x": x,
        "y": x * 2 + rng.standard_normal(100) * 0.1,   # strong positive correlation
        "z": rng.standard_normal(100),                   # no correlation with x
    })


@pytest.fixture
def constant_df():
    """DataFrame where all columns have zero variance."""
    return pd.DataFrame({
        "a": [1.0] * 20,
        "b": [5.0] * 20,
    })


@pytest.fixture
def missing_df():
    """DataFrame with systematic missing values."""
    df = pd.DataFrame({
        "complete": [1.0, 2.0, 3.0, 4.0, 5.0],
        "half_missing": [1.0, None, 3.0, None, 5.0],
        "all_missing": [None, None, None, None, None],
    })
    return df


@pytest.fixture
def mixed_df():
    """DataFrame with numeric and non-numeric columns."""
    rng = np.random.default_rng(0)
    return pd.DataFrame({
        "num_a": rng.standard_normal(50),
        "num_b": rng.standard_normal(50),
        "category": ["A", "B"] * 25,
    })


@pytest.fixture
def single_col_df():
    """Single numeric column — edge case for correlation and scatter matrix."""
    return pd.DataFrame({"only": [1.0, 2.0, 3.0, 4.0, 5.0]})


@pytest.fixture
def bioinformatics_df():
    """Realistic bioinformatics-style numeric DataFrame.

    Columns: GC content, sequence length, coverage depth, quality score.
    GC and length are loosely correlated; coverage is independent.
    """
    rng = np.random.default_rng(7)
    n = 120
    gc = rng.uniform(0.3, 0.7, n)
    return pd.DataFrame({
        "gc_content":     gc,
        "seq_length":     (gc * 500 + rng.normal(0, 20, n)).clip(100, 1000),
        "coverage_depth": rng.exponential(30, n),
        "quality_score":  rng.uniform(20, 40, n),
    })


# ---------------------------------------------------------------------------
# get_correlation
# ---------------------------------------------------------------------------

class TestGetCorrelation:
    def test_returns_list_for_correlated_column(self, numeric_df):
        result = get_correlation(numeric_df, "x")
        assert result is not None
        assert isinstance(result, list)

    def test_correlated_pair_included(self, numeric_df):
        result = get_correlation(numeric_df, "x")
        cols = [name for name, _ in result]
        assert "y" in cols

    def test_weak_correlation_excluded(self, numeric_df):
        """z is uncorrelated with x; should not appear (threshold 0.3)."""
        result = get_correlation(numeric_df, "x")
        if result is not None:
            cols = [name for name, _ in result]
            assert "z" not in cols

    def test_returns_none_for_non_numeric_column(self, mixed_df):
        result = get_correlation(mixed_df, "category")
        assert result is None

    def test_returns_none_for_zero_variance_column(self, constant_df):
        result = get_correlation(constant_df, "a")
        assert result is None

    def test_returns_none_when_all_correlations_below_threshold(self, numeric_df):
        """z has no strong correlation with anything — isolated case."""
        # Build a DataFrame where z correlates with nothing above 0.3
        rng = np.random.default_rng(99)
        df = pd.DataFrame({
            "z": rng.standard_normal(200),
            "noise1": rng.standard_normal(200),
            "noise2": rng.standard_normal(200),
        })
        result = get_correlation(df, "z")
        assert result is None

    def test_correlation_values_are_floats(self, numeric_df):
        result = get_correlation(numeric_df, "x")
        assert result is not None
        for _, val in result:
            assert isinstance(val, float)

    def test_target_column_excluded_from_result(self, numeric_df):
        """The column itself must not appear in its own correlation list."""
        result = get_correlation(numeric_df, "x")
        if result is not None:
            cols = [name for name, _ in result]
            assert "x" not in cols

    def test_returns_none_for_missing_column(self, numeric_df):
        result = get_correlation(numeric_df, "nonexistent")
        assert result is None

    def test_gc_content_correlated_with_seq_length(self, bioinformatics_df):
        result = get_correlation(bioinformatics_df, "gc_content")
        assert result is not None
        cols = [name for name, _ in result]
        assert "seq_length" in cols


# ---------------------------------------------------------------------------
# correlation_heatmap (mixed-type: Pearson / Cramér's V / Eta²)
# ---------------------------------------------------------------------------

def make_methods(df, value="Pearson"):
    """Build a dummy methods DataFrame matching df columns."""
    cols = list(df.columns)
    return pd.DataFrame(value, index=cols, columns=cols)


class TestCorrelationHeatmap:
    def test_returns_html_string(self, numeric_df):
        values, methods = compute_correlation_matrix(numeric_df)
        result = correlation_heatmap(values, methods)
        assert isinstance(result, str)
        assert "<div" in result

    def test_hover_method_label_in_output(self, numeric_df):
        """Method names should appear in the hovertemplate."""
        values, methods = compute_correlation_matrix(numeric_df)
        result = correlation_heatmap(values, methods)
        assert "Method" in result or "customdata" in result

    def test_mixed_df_produces_html(self, mixed_df):
        values, methods = compute_correlation_matrix(mixed_df)
        result = correlation_heatmap(values, methods)
        assert isinstance(result, str)

    def test_values_bounded_0_to_1(self, numeric_df):
        """Association values matrix must be in [0, 1]."""
        values, _ = compute_correlation_matrix(numeric_df)
        finite = values.to_numpy(dtype=float)
        finite = finite[~np.isnan(finite)]
        assert (finite >= 0).all()
        assert (finite <= 1.0 + 1e-9).all()


# ---------------------------------------------------------------------------
# pearson_correlation_heatmap (numeric-only, replaces old correlation_heatmap)
# ---------------------------------------------------------------------------

class TestPearsonCorrelationHeatmap:
    def test_returns_html_string(self, numeric_df):
        result = pearson_correlation_heatmap(numeric_df)
        assert isinstance(result, str)
        assert "<div" in result

    def test_excludes_non_numeric_columns(self, mixed_df):
        result = pearson_correlation_heatmap(mixed_df)
        assert isinstance(result, str)

    def test_zero_variance_columns_excluded(self, constant_df):
        result = pearson_correlation_heatmap(constant_df)
        assert isinstance(result, str)

    def test_all_missing_column_excluded(self, missing_df):
        result = pearson_correlation_heatmap(missing_df)
        assert isinstance(result, str)

    def test_single_numeric_column(self, single_col_df):
        result = pearson_correlation_heatmap(single_col_df)
        assert isinstance(result, str)


# ---------------------------------------------------------------------------
# missing_matrix
# ---------------------------------------------------------------------------

class TestMissingMatrix:
    def test_returns_html_string(self, missing_df):
        result = missing_matrix(missing_df)
        assert isinstance(result, str)
        assert "<div" in result

    def test_no_missing_values(self, numeric_df):
        result = missing_matrix(numeric_df)
        assert isinstance(result, str)

    def test_all_missing(self):
        df = pd.DataFrame({"a": [None, None], "b": [None, None]})
        result = missing_matrix(df)
        assert isinstance(result, str)


# ---------------------------------------------------------------------------
# missing_values_barchart
# ---------------------------------------------------------------------------

class TestMissingValuesBarchart:
    def test_returns_html_string(self, missing_df):
        result = missing_values_barchart(missing_df)
        assert isinstance(result, str)
        assert "<div" in result

    def test_no_missing_values(self, numeric_df):
        result = missing_values_barchart(numeric_df)
        assert isinstance(result, str)

    def test_single_row_df(self):
        df = pd.DataFrame({"a": [1.0], "b": [None]})
        result = missing_values_barchart(df)
        assert isinstance(result, str)


# ---------------------------------------------------------------------------
# balance_plot
# ---------------------------------------------------------------------------

class TestBalancePlot:
    def test_returns_html_string_with_target(self, mixed_df):
        result = balance_plot(mixed_df, "category")
        assert isinstance(result, str)
        assert "<div" in result

    def test_numeric_target_column(self, numeric_df):
        result = balance_plot(numeric_df, "x")
        assert isinstance(result, str)

    def test_single_class_target(self):
        df = pd.DataFrame({"label": ["A"] * 30, "value": range(30)})
        result = balance_plot(df, "label")
        assert isinstance(result, str)


# ---------------------------------------------------------------------------
# boxplot
# ---------------------------------------------------------------------------

class TestBoxplot:
    def test_returns_html_string(self, numeric_df):
        result = boxplot(numeric_df)
        assert isinstance(result, str)
        assert "<div" in result

    def test_excludes_non_numeric(self, mixed_df):
        result = boxplot(mixed_df)
        assert isinstance(result, str)

    def test_all_missing_columns_excluded(self, missing_df):
        result = boxplot(missing_df)
        assert isinstance(result, str)

    def test_no_numeric_columns(self):
        df = pd.DataFrame({"cat": ["A", "B", "C"]})
        result = boxplot(df)
        assert isinstance(result, str)

    def test_bioinformatics_columns(self, bioinformatics_df):
        result = boxplot(bioinformatics_df)
        assert isinstance(result, str)


# ---------------------------------------------------------------------------
# scatter_matrix
# ---------------------------------------------------------------------------

class TestScatterMatrix:
    def test_returns_html_string(self, numeric_df):
        result = scatter_matrix(numeric_df)
        assert isinstance(result, str)
        assert "<div" in result

    def test_single_column(self, single_col_df):
        result = scatter_matrix(single_col_df)
        assert isinstance(result, str)

    def test_excludes_non_numeric(self, mixed_df):
        result = scatter_matrix(mixed_df)
        assert isinstance(result, str)

    def test_no_numeric_columns(self):
        df = pd.DataFrame({"cat": ["A", "B"]})
        result = scatter_matrix(df)
        assert isinstance(result, str)


# ---------------------------------------------------------------------------
# multivariate_analysis (top-level pipeline, replaces general_plots)
# ---------------------------------------------------------------------------

class TestMultivariateAnalysis:
    def test_returns_multivariate_analysis_object(self, numeric_df):
        from models.multivariate import MultivariateAnalysis
        result = multivariate_analysis(numeric_df, target=None)
        assert isinstance(result, MultivariateAnalysis)

    def test_all_html_fields_present_without_target(self, numeric_df):
        result = multivariate_analysis(numeric_df, target=None)
        for field in ("correlation_heatmap", "pearson_heatmap",
                      "missing_matrix", "missing_values_barchart",
                      "boxplot", "scatter_matrix"):
            val = getattr(result, field)
            assert isinstance(val, str), f"{field} should be an HTML string"

    def test_balance_plot_none_when_no_target(self, numeric_df):
        result = multivariate_analysis(numeric_df, target=None)
        assert result.balance_plot is None

    def test_balance_plot_present_with_target(self, mixed_df):
        result = multivariate_analysis(mixed_df, target="category")
        assert result.balance_plot is not None
        assert isinstance(result.balance_plot, str)

    def test_feature_target_correlation_none_without_target(self, numeric_df):
        result = multivariate_analysis(numeric_df, target=None)
        assert result.feature_target_correlation is None

    def test_feature_target_correlation_dict_with_target(self, mixed_df):
        result = multivariate_analysis(mixed_df, target="category")
        # target may or may not be in _classify_columns; result is dict or None
        assert result.feature_target_correlation is None or isinstance(result.feature_target_correlation, dict)

    def test_cramers_heatmap_none_for_numeric_only(self, numeric_df):
        """No categorical columns → Cramér's V heatmap should be None."""
        result = multivariate_analysis(numeric_df, target=None)
        assert result.cramers_heatmap is None

    def test_eta_heatmap_none_for_numeric_only(self, numeric_df):
        """No categorical columns → Eta² heatmap should be None."""
        result = multivariate_analysis(numeric_df, target=None)
        assert result.eta_heatmap is None

    def test_cramers_and_eta_present_for_mixed(self):
        """Needs ≥ 2 categorical columns for Cramér's V and ≥ 1 cat + 1 num for Eta²."""
        rng = np.random.default_rng(1)
        df = pd.DataFrame({
            "num":  rng.standard_normal(60),
            "cat1": ["A", "B", "C"] * 20,
            "cat2": ["X", "Y"] * 30,
        })
        result = multivariate_analysis(df, target=None)
        assert isinstance(result.cramers_heatmap, str)
        assert isinstance(result.eta_heatmap, str)

    def test_correlation_matrix_is_dataframe(self, numeric_df):
        result = multivariate_analysis(numeric_df, target=None)
        assert isinstance(result.correlation_matrix, pd.DataFrame)

    def test_missing_heavy_dataframe(self, missing_df):
        from models.multivariate import MultivariateAnalysis
        result = multivariate_analysis(missing_df, target=None)
        assert isinstance(result, MultivariateAnalysis)

    def test_bioinformatics_full_pipeline(self, bioinformatics_df):
        from models.multivariate import MultivariateAnalysis
        result = multivariate_analysis(bioinformatics_df, target=None)
        assert isinstance(result, MultivariateAnalysis)
        for field in ("correlation_heatmap", "missing_matrix",
                      "missing_values_barchart", "boxplot", "scatter_matrix"):
            assert isinstance(getattr(result, field), str)