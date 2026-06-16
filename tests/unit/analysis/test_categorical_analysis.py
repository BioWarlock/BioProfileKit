import numpy as np
import pandas as pd
import pytest
import sys
import os

# ---------------------------------------------------------------------------
# Path setup – adjust if your project root differs
# ---------------------------------------------------------------------------
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from analysis.categorical_analysis import categorical_columns


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def balanced_df():
    """Equal counts for three categories – clean, no NaN."""
    return pd.DataFrame({"col": ["A"] * 10 + ["B"] * 10 + ["C"] * 10})


@pytest.fixture
def imbalanced_df():
    """Heavily skewed: one dominant category."""
    return pd.DataFrame({"col": ["A"] * 95 + ["B"] * 3 + ["C"] * 2})


@pytest.fixture
def single_category_df():
    """Only one unique value – triggers cib_ratio division-by-zero risk."""
    return pd.DataFrame({"col": ["A"] * 20})


@pytest.fixture
def df_with_nan():
    """Mix of valid values and NaN."""
    return pd.DataFrame({"col": ["A"] * 10 + ["B"] * 10 + [None] * 5})


@pytest.fixture
def all_nan_df():
    """All values are NaN – n == 0 path."""
    return pd.DataFrame({"col": [None, None, None]})


# --- Domain-specific fixtures -----------------------------------------------

@pytest.fixture
def organism_df():
    """Realistic taxonomic label column with natural imbalance."""
    return pd.DataFrame({"organism": (
        ["Escherichia coli"] * 40 +
        ["Homo sapiens"] * 30 +
        ["Bacillus subtilis"] * 15 +
        ["Mus musculus"] * 10 +
        ["unknown"] * 5
    )})


@pytest.fixture
def cog_df():
    """COG functional category column – single-letter codes, known vocabulary."""
    # J=Translation, K=Transcription, L=Replication, rare: U, W (1 each)
    return pd.DataFrame({"cog": (
        ["J"] * 50 +
        ["K"] * 30 +
        ["L"] * 18 +
        ["U"] * 1 +
        ["W"] * 1
    )})


@pytest.fixture
def go_term_df():
    """GO term column – long, structured strings with mixed NaN."""
    terms = (
        ["GO:0008150"] * 25 +   # biological_process
        ["GO:0003674"] * 25 +   # molecular_function
        ["GO:0005575"] * 20 +   # cellular_component
        ["GO:0006412"] * 15 +   # translation
        ["GO:0003735"] * 10 +   # ribosome structural constituent
        [None] * 5
    )
    return pd.DataFrame({"go_term": terms})


# ---------------------------------------------------------------------------
# Basic return type and field presence
# ---------------------------------------------------------------------------

class TestReturnType:
    def test_returns_categorical_columns_object(self, balanced_df):
        result = categorical_columns(balanced_df, "col")
        # Duck-type check: all expected attributes exist
        for attr in (
            "name", "unique_categories", "mode", "entropy", "frequencies",
            "gini", "simpson_diversity", "value_counts", "max_category_length",
            "min_category_length", "cardinality_ratio", "rare_categories",
            "top_5_coverage", "cib_ratio", "top_1_coverage",
            "effective_cardinality", "memory",
        ):
            assert hasattr(result, attr), f"Missing attribute: {attr}"

    def test_name_matches_column(self, balanced_df):
        result = categorical_columns(balanced_df, "col")
        assert result.name == "col"


# ---------------------------------------------------------------------------
# Unique categories / cardinality
# ---------------------------------------------------------------------------

class TestCardinality:
    def test_unique_categories_balanced(self, balanced_df):
        result = categorical_columns(balanced_df, "col")
        assert result.unique_categories == 3

    def test_unique_categories_single(self, single_category_df):
        result = categorical_columns(single_category_df, "col")
        assert result.unique_categories == 1

    def test_effective_cardinality_range(self, balanced_df):
        result = categorical_columns(balanced_df, "col")
        assert 0 < result.effective_cardinality <= 100

    def test_cardinality_ratio_balanced(self, balanced_df):
        result = categorical_columns(balanced_df, "col")
        # 3 unique / 30 total = 0.1
        assert result.cardinality_ratio == pytest.approx(0.1, abs=1e-3)


# ---------------------------------------------------------------------------
# Diversity metrics: entropy, Gini, Simpson
# ---------------------------------------------------------------------------

class TestDiversityMetrics:
    def test_entropy_maximised_for_balanced(self, balanced_df):
        """Balanced distribution should yield maximum entropy for 3 classes ≈ log2(3)."""
        result = categorical_columns(balanced_df, "col")
        max_entropy = np.log2(3)
        assert result.entropy == pytest.approx(max_entropy, abs=0.01)

    def test_entropy_zero_for_single_category(self, single_category_df):
        result = categorical_columns(single_category_df, "col")
        assert result.entropy == pytest.approx(0.0, abs=0.01)

    def test_gini_zero_for_single_category(self, single_category_df):
        result = categorical_columns(single_category_df, "col")
        assert result.gini == pytest.approx(0.0, abs=0.01)

    def test_gini_bounded(self, balanced_df):
        result = categorical_columns(balanced_df, "col")
        assert 0.0 <= result.gini < 1.0

    def test_simpson_at_least_one(self, balanced_df):
        """Simpson diversity index (1/sum(p²)) is ≥ 1."""
        result = categorical_columns(balanced_df, "col")
        assert result.simpson_diversity >= 1.0

    def test_simpson_equals_one_for_single_category(self, single_category_df):
        result = categorical_columns(single_category_df, "col")
        assert result.simpson_diversity == pytest.approx(1.0, abs=0.01)

    def test_metrics_are_nan_for_all_nan_input(self, all_nan_df):
        result = categorical_columns(all_nan_df, "col")
        assert np.isnan(result.entropy)
        assert np.isnan(result.gini)
        assert np.isnan(result.simpson_diversity)
        assert np.isnan(result.cardinality_ratio)


# ---------------------------------------------------------------------------
# Coverage metrics
# ---------------------------------------------------------------------------

class TestCoverage:
    def test_top_1_coverage_single_category(self, single_category_df):
        result = categorical_columns(single_category_df, "col")
        assert result.top_1_coverage == pytest.approx(1.0, abs=1e-6)

    def test_top_1_coverage_balanced(self, balanced_df):
        result = categorical_columns(balanced_df, "col")
        assert result.top_1_coverage == pytest.approx(1 / 3, abs=0.01)

    def test_top_5_coverage_at_most_one(self, balanced_df):
        result = categorical_columns(balanced_df, "col")
        assert result.top_5_coverage <= 1.0

    def test_top_5_coverage_dominated(self, imbalanced_df):
        result = categorical_columns(imbalanced_df, "col")
        # A alone covers 95 %, so top-5 must be ≥ 0.95
        assert result.top_5_coverage >= 0.95


# ---------------------------------------------------------------------------
# CIB ratio (class imbalance)
# ---------------------------------------------------------------------------

class TestCIBRatio:
    def test_cib_ratio_balanced_equals_one(self, balanced_df):
        result = categorical_columns(balanced_df, "col")
        assert result.cib_ratio == pytest.approx(1.0, abs=1e-6)

    def test_cib_ratio_imbalanced(self, imbalanced_df):
        result = categorical_columns(imbalanced_df, "col")
        # 95 / 2 = 47.5
        assert result.cib_ratio == pytest.approx(47.5, abs=0.1)

    def test_cib_ratio_single_category(self, single_category_df):
        """
        Single category: min == max, so cib_ratio == 1.
        NOTE: current implementation returns 1.0 here because
        value_counts.max() / value_counts.min() == n / n.
        This test documents the expected behaviour.
        """
        result = categorical_columns(single_category_df, "col")
        assert result.cib_ratio == pytest.approx(1.0, abs=1e-6)


# ---------------------------------------------------------------------------
# Rare categories
# ---------------------------------------------------------------------------

class TestRareCategories:
    def test_no_rare_in_balanced(self, balanced_df):
        """Each category is 33 % of 30 rows – well above 1 % threshold."""
        result = categorical_columns(balanced_df, "col")
        assert result.rare_categories == 0

    def test_rare_detected_in_imbalanced(self, imbalanced_df):
        """B (3 %) and C (2 %) are below 1 % of 100 rows only if < 1.
        With threshold len(df)*0.01 = 1.0: counts of 2 and 3 are NOT rare.
        Adjust expectation to match current implementation."""
        result = categorical_columns(imbalanced_df, "col")
        # count < 1.0 → no category qualifies; documents actual behaviour
        assert result.rare_categories == 0

    def test_rare_with_singleton(self):
        """A category appearing exactly once in a 200-row dataset is rare."""
        data = ["A"] * 198 + ["B"] * 1 + ["C"] * 1
        df = pd.DataFrame({"col": data})
        result = categorical_columns(df, "col")
        # threshold = 200 * 0.01 = 2.0 → counts of 1 are < 2 → rare
        assert result.rare_categories == 2


# ---------------------------------------------------------------------------
# NaN handling
# ---------------------------------------------------------------------------

class TestNaNHandling:
    def test_nan_excluded_from_n(self, df_with_nan):
        """n should count only non-null values (20), not total rows (25)."""
        result = categorical_columns(df_with_nan, "col")
        # With n=20 and 2 categories, cardinality_ratio = 2/20 = 0.1
        assert result.cardinality_ratio == pytest.approx(0.1, abs=1e-3)

    def test_mode_valid_with_nan(self, df_with_nan):
        result = categorical_columns(df_with_nan, "col")
        assert result.mode in ("A", "B")

    def test_mode_empty_string_for_all_nan(self, all_nan_df):
        result = categorical_columns(all_nan_df, "col")
        assert result.mode == ""


# ---------------------------------------------------------------------------
# String length metrics
# ---------------------------------------------------------------------------

class TestStringLengths:
    def test_length_metrics_present(self, balanced_df):
        result = categorical_columns(balanced_df, "col")
        assert result.min_category_length >= 1
        assert result.max_category_length >= result.min_category_length

    def test_length_varies_with_input(self):
        df = pd.DataFrame({"col": ["A", "BB", "CCC"]})
        result = categorical_columns(df, "col")
        assert result.min_category_length == 1
        assert result.max_category_length == 3


# ---------------------------------------------------------------------------
# frequencies / value_counts dicts capped at 20
# ---------------------------------------------------------------------------

class TestOutputDicts:
    def test_frequencies_capped_at_20(self):
        cats = [str(i) for i in range(50)]
        df = pd.DataFrame({"col": cats * 2})
        result = categorical_columns(df, "col")
        assert len(result.frequencies) <= 20

    def test_value_counts_capped_at_20(self):
        cats = [str(i) for i in range(50)]
        df = pd.DataFrame({"col": cats * 2})
        result = categorical_columns(df, "col")
        assert len(result.value_counts) <= 20

    def test_frequencies_sum_to_at_most_one(self, balanced_df):
        result = categorical_columns(balanced_df, "col")
        assert sum(result.frequencies.values()) == pytest.approx(1.0, abs=1e-6)


# ---------------------------------------------------------------------------
# Memory usage
# ---------------------------------------------------------------------------

class TestMemory:
    def test_memory_is_positive_integer(self, balanced_df):
        result = categorical_columns(balanced_df, "col")
        assert isinstance(result.memory, int)
        assert result.memory > 0


# ---------------------------------------------------------------------------
# Domain-specific: taxonomic organism labels
# ---------------------------------------------------------------------------

class TestOrganismColumn:
    def test_unique_category_count(self, organism_df):
        result = categorical_columns(organism_df, "organism")
        assert result.unique_categories == 5

    def test_mode_is_dominant_organism(self, organism_df):
        result = categorical_columns(organism_df, "organism")
        assert result.mode == "Escherichia coli"

    def test_string_lengths_reflect_long_names(self, organism_df):
        result = categorical_columns(organism_df, "organism")
        # "Bacillus subtilis" = 17, "unknown" = 7
        assert result.min_category_length == len("unknown")
        assert result.max_category_length == len("Bacillus subtilis")

    def test_imbalance_ratio_organism(self, organism_df):
        result = categorical_columns(organism_df, "organism")
        # max=40 (E. coli), min=5 (unknown) → 8.0
        assert result.cib_ratio == pytest.approx(8.0, abs=0.01)

    def test_top_1_coverage_organism(self, organism_df):
        result = categorical_columns(organism_df, "organism")
        # E. coli: 40/100 = 0.4
        assert result.top_1_coverage == pytest.approx(0.4, abs=0.01)

    def test_top_5_coverage_all_captured(self, organism_df):
        result = categorical_columns(organism_df, "organism")
        # Only 5 categories → top-5 covers everything
        assert result.top_5_coverage == pytest.approx(1.0, abs=1e-6)

    def test_entropy_lower_than_maximum(self, organism_df):
        result = categorical_columns(organism_df, "organism")
        max_entropy = np.log2(5)
        # Distribution is skewed, so entropy < max
        assert result.entropy < max_entropy

    def test_no_rare_categories_organism(self, organism_df):
        # threshold = 100 * 0.01 = 1.0 → min count is 5 → no rare
        result = categorical_columns(organism_df, "organism")
        assert result.rare_categories == 0


# ---------------------------------------------------------------------------
# Domain-specific: COG functional categories
# ---------------------------------------------------------------------------

class TestCOGColumn:
    def test_single_letter_length(self, cog_df):
        result = categorical_columns(cog_df, "cog")
        assert result.min_category_length == 1
        assert result.max_category_length == 1

    def test_mode_is_most_frequent_cog(self, cog_df):
        result = categorical_columns(cog_df, "cog")
        assert result.mode == "J"

    def test_rare_singleton_cog_categories(self, cog_df):
        result = categorical_columns(cog_df, "cog")
        # threshold = 100 * 0.01 = 1.0 → U and W (count=1) are rare
        assert result.rare_categories == 2

    def test_cib_ratio_cog(self, cog_df):
        result = categorical_columns(cog_df, "cog")
        # max=50 (J), min=1 (U or W) → 50.0
        assert result.cib_ratio == pytest.approx(50.0, abs=0.01)

    def test_top_1_coverage_cog(self, cog_df):
        result = categorical_columns(cog_df, "cog")
        assert result.top_1_coverage == pytest.approx(0.5, abs=0.01)

    def test_gini_high_for_skewed_cog(self, cog_df):
        result = categorical_columns(cog_df, "cog")
        # Skewed distribution → Gini closer to 1 than to 0
        assert result.gini > 0.5


# ---------------------------------------------------------------------------
# Domain-specific: GO term column with NaN
# ---------------------------------------------------------------------------

class TestGOTermColumn:
    def test_nan_excluded_from_metrics(self, go_term_df):
        result = categorical_columns(go_term_df, "go_term")
        # n = 95 (5 NaN excluded); 5 unique terms → ratio = 5/95
        assert result.cardinality_ratio == pytest.approx(5 / 95, abs=1e-3)

    def test_go_term_string_length(self, go_term_df):
        result = categorical_columns(go_term_df, "go_term")
        # All GO terms follow "GO:XXXXXXX" = 10 chars
        assert result.min_category_length == 10
        assert result.max_category_length == 10

    def test_mode_is_most_frequent_go_term(self, go_term_df):
        result = categorical_columns(go_term_df, "go_term")
        assert result.mode in ("GO:0008150", "GO:0003674")

    def test_no_rare_go_terms(self, go_term_df):
        result = categorical_columns(go_term_df, "go_term")
        # threshold = 100 * 0.01 = 1.0; min count is 10 → no rare
        assert result.rare_categories == 0

    def test_frequencies_exclude_nan(self, go_term_df):
        result = categorical_columns(go_term_df, "go_term")
        assert all(k is not None for k in result.frequencies)
        assert sum(result.frequencies.values()) == pytest.approx(1.0, abs=1e-6)