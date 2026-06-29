"""
Unit tests for cli/report_json.py

Strategy: all objects passed to report_json functions are MagicMocks or
simple dataclass instances — no filesystem, no pandas, no numpy required
for most tests. Where numpy types need to be exercised, small np arrays
are constructed directly.
"""
import json
import math
import sys
import os
import pytest
import numpy as np
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from cli.report_json import (
    _num, _present, _listify, _summary, _outliers,
    _numeric_block, _categorical_block, _dna_block, _protein_block,
    _taxonomy_block, _measurement_block, _correlated_partners,
    _role, _column_entry, _quality_block, build_result,
    write_result_json, _json_default, CORR_THRESHOLD,
)


# ---------------------------------------------------------------------------
# _num
# ---------------------------------------------------------------------------

class TestNum:
    def test_none_returns_none(self):
        assert _num(None) is None

    def test_np_integer(self):
        result = _num(np.int64(42))
        assert result == 42
        assert isinstance(result, int)

    def test_np_floating_finite(self):
        result = _num(np.float64(3.14))
        assert abs(result - 3.14) < 1e-6
        assert isinstance(result, float)

    def test_np_floating_nan_returns_none(self):
        assert _num(np.float64(float("nan"))) is None

    def test_np_floating_inf_returns_none(self):
        assert _num(np.float64(float("inf"))) is None

    def test_python_float_inf_returns_none(self):
        assert _num(float("inf")) is None

    def test_python_int_passthrough(self):
        assert _num(7) == 7

    def test_python_bool_passthrough(self):
        assert _num(True) is True

    def test_python_str_passthrough(self):
        assert _num("hello") == "hello"

    def test_np_ndarray(self):
        arr = np.array([1, 2, 3])
        result = _num(arr)
        assert result == [1, 2, 3]

    def test_plain_float(self):
        assert _num(1.5) == 1.5

    def test_unknown_type_passthrough(self):
        obj = object()
        assert _num(obj) is obj


# ---------------------------------------------------------------------------
# _present
# ---------------------------------------------------------------------------

class TestPresent:
    def test_none_is_not_present(self):
        assert _present(None) is False

    def test_empty_list(self):
        assert _present([]) is False

    def test_nonempty_list(self):
        assert _present([1]) is True

    def test_empty_string(self):
        assert _present("") is False

    def test_nonempty_string(self):
        assert _present("a") is True

    def test_empty_dict(self):
        assert _present({}) is False

    def test_nonempty_dict(self):
        assert _present({"k": 1}) is True

    def test_empty_ndarray(self):
        assert _present(np.array([])) is False

    def test_nonempty_ndarray(self):
        assert _present(np.array([1])) is True

    def test_empty_set(self):
        assert _present(set()) is False

    def test_nonempty_set(self):
        assert _present({1}) is True

    def test_pandas_series(self):
        import pandas as pd
        assert _present(pd.Series([], dtype=float)) is False
        assert _present(pd.Series([1])) is True

    def test_zero_is_falsy(self):
        assert _present(0) is False

    def test_one_is_truthy(self):
        assert _present(1) is True


# ---------------------------------------------------------------------------
# _listify
# ---------------------------------------------------------------------------

class TestListify:
    def test_none_returns_none(self):
        assert _listify(None) is None

    def test_empty_list_returns_none(self):
        assert _listify([]) is None

    def test_list_of_ints(self):
        assert _listify([1, 2, 3]) == [1, 2, 3]

    def test_tuple(self):
        assert _listify((1, 2)) == [1, 2]

    def test_set_returns_list(self):
        result = _listify({42})
        assert result == [42]

    def test_ndarray(self):
        result = _listify(np.array([1, 2, 3]))
        assert result == [1, 2, 3]

    def test_scalar_value(self):
        assert _listify(5) == 5

    def test_pandas_series(self):
        import pandas as pd
        result = _listify(pd.Series([10, 20]))
        assert result == [10, 20]


# ---------------------------------------------------------------------------
# _summary
# ---------------------------------------------------------------------------

class TestSummary:
    def test_none_returns_none(self):
        assert _summary(None) is None

    def test_summary_with_mock(self):
        s = MagicMock()
        s.min = 1.0
        s.max = 5.0
        s.mean = 3.0
        result = _summary(s)
        assert result == {"min": 1.0, "max": 5.0, "mean": 3.0}


# ---------------------------------------------------------------------------
# _outliers
# ---------------------------------------------------------------------------

class TestOutliers:
    def test_none_returns_none(self):
        assert _outliers(None) is None

    def test_outlier_block(self):
        o = MagicMock()
        o.lower_bound = 0.1
        o.upper_bound = 9.9
        o.medcouple = 0.05
        o.n_lower_iqr = 2
        o.n_upper_iqr = 3
        o.n_lower_mzscore = 1
        o.n_upper_mzscore = 1
        o.n_lower_zscore = 0
        o.n_upper_zscore = 2
        result = _outliers(o)
        assert result["lower_bound"] == 0.1
        assert result["n_upper_iqr"] == 3
        assert set(result.keys()) == {
            "lower_bound", "upper_bound", "medcouple",
            "n_lower_iqr", "n_upper_iqr",
            "n_lower_mzscore", "n_upper_mzscore",
            "n_lower_zscore", "n_upper_zscore",
        }


# ---------------------------------------------------------------------------
# _numeric_block / _categorical_block
# ---------------------------------------------------------------------------

class TestNumericBlock:
    def test_none_returns_none(self):
        assert _numeric_block(None) is None

    def test_produces_expected_keys(self):
        n = MagicMock()
        n.min = 0.0; n.max = 10.0; n.mean = 5.0; n.median = 5.0
        n.mode = 3.0; n.std = 2.0; n.sum = 50.0; n.kurtosis = 0.1
        n.skewness = 0.2; n.coefficient_of_variation = 0.4; n.mad = 1.5
        n.quantiles = [1.0, 5.0, 9.0]; n.infinity = 0
        n.negative_count = 0; n.zero_count = 1; n.outliers = None
        result = _numeric_block(n)
        assert result["min"] == 0.0
        assert result["outliers"] is None


class TestCategoricalBlock:
    def test_none_returns_none(self):
        assert _categorical_block(None) is None

    def test_produces_expected_keys(self):
        c = MagicMock()
        c.unique_categories = 5; c.mode = "cat_a"
        c.entropy = 1.2; c.gini = 0.3; c.simpson_diversity = 0.7
        c.max_category_length = 10; c.min_category_length = 2
        c.cardinality_ratio = 0.5; c.rare_categories = 1
        c.top_5_coverage = 0.8; c.top_1_coverage = 0.4
        c.cib_ratio = 0.6; c.effective_cardinality = 3.0
        result = _categorical_block(c)
        assert result["mode"] == "cat_a"
        assert result["unique_categories"] == 5


# ---------------------------------------------------------------------------
# _dna_block / _protein_block
# ---------------------------------------------------------------------------

class TestDnaBlock:
    def _mock_summary(self, val=1.0):
        s = MagicMock(); s.min = val; s.max = val; s.mean = val
        return s

    def test_produces_expected_keys(self):
        d = MagicMock()
        for attr in ["gc_content", "ambiguous_base_ratio", "length_stats",
                     "codon_completeness", "gc_skew", "at_skew",
                     "cpg_observed_expected", "tpa_observed_expected", "low_complexity"]:
            setattr(d, attr, self._mock_summary())
        d.length_outliers = None
        d.reverse_complement_ratio = 5.0
        result = _dna_block(d)
        assert "gc_content" in result
        assert result["reverse_complement_ratio"] == 5.0
        assert result["length_outliers"] is None


class TestProteinBlock:
    def _mock_summary(self, val=1.0):
        s = MagicMock(); s.min = val; s.max = val; s.mean = val
        return s

    def test_produces_expected_keys(self):
        p = MagicMock()
        for attr in ["ambiguous_residue_ratio", "length_stats", "length_outliers",
                     "low_complexity", "gravy", "cysteine_count", "disorder_propensity"]:
            setattr(p, attr, self._mock_summary() if attr != "length_outliers" else None)
        p.stop_codon_ratio = 0.5
        p.aa_group_distribution = {"Unpolar": 0.4, "Aromatic": 0.1}
        result = _protein_block(p)
        assert result["stop_codon_ratio"] == 0.5
        assert result["aa_group_distribution"]["Unpolar"] == 0.4

    def test_none_aa_group(self):
        p = MagicMock()
        p.aa_group_distribution = None
        p.stop_codon_ratio = 0.0
        for attr in ["ambiguous_residue_ratio", "length_stats", "length_outliers",
                     "low_complexity", "gravy", "cysteine_count", "disorder_propensity"]:
            setattr(p, attr, None)
        result = _protein_block(p)
        assert result["aa_group_distribution"] is None


# ---------------------------------------------------------------------------
# _taxonomy_block
# ---------------------------------------------------------------------------

class TestTaxonomyBlock:
    def test_none_returns_none(self):
        assert _taxonomy_block(None) is None

    def test_not_taxonomy_returns_none(self):
        t = MagicMock()
        t.is_taxonomy = False
        assert _taxonomy_block(t) is None

    def test_basic_taxonomy(self):
        t = MagicMock()
        t.is_taxonomy = True
        t.is_mixed = False
        t.rank_distribution = {"species": 0.9}
        t.outdated_names = {"OldName": "NewName"}
        t.invalid_names = ["BadName"]
        result = _taxonomy_block(t)
        assert result["is_taxonomy"] is True
        assert result["rank_distribution"] == {"species": 0.9}
        assert result["name_corrections"] == {"OldName": "NewName"}
        assert result["invalid_values"] == ["BadName"]

    def test_empty_optional_fields_omitted(self):
        t = MagicMock()
        t.is_taxonomy = True
        t.is_mixed = False
        t.rank_distribution = None
        t.outdated_names = None
        t.invalid_names = None
        result = _taxonomy_block(t)
        assert "rank_distribution" not in result
        assert "name_corrections" not in result
        assert "invalid_values" not in result


# ---------------------------------------------------------------------------
# _measurement_block
# ---------------------------------------------------------------------------

class TestMeasurementBlock:
    def test_none_returns_none(self):
        assert _measurement_block(None) is None

    def test_basic_measurement(self):
        m = MagicMock()
        m.units = ["mg/L"]
        m.unit_counts = {"mg/L": 10}
        m.with_measurement = True
        result = _measurement_block(m)
        assert result["units"] == ["mg/L"]
        assert result["with_measurement"] is True
        assert result["unit_counts"] == {"mg/L": 10}


# ---------------------------------------------------------------------------
# _correlated_partners
# ---------------------------------------------------------------------------

class TestCorrelatedPartners:
    def test_empty_associations(self):
        assert _correlated_partners("a", None, 0.9) == []
        assert _correlated_partners("a", [], 0.9) == []

    def test_below_threshold_excluded(self):
        pairs = [{"var1": "a", "var2": "b", "value": 0.5, "method": "pearson"}]
        result = _correlated_partners("a", pairs, 0.9)
        assert result == []

    def test_above_threshold_included(self):
        pairs = [{"var1": "a", "var2": "b", "value": 0.95, "method": "pearson"}]
        result = _correlated_partners("a", pairs, 0.9)
        assert len(result) == 1
        assert result[0]["column"] == "b"
        assert result[0]["value"] == 0.95

    def test_keeps_best_per_partner(self):
        pairs = [
            {"var1": "a", "var2": "b", "value": 0.91, "method": "pearson"},
            {"var1": "a", "var2": "b", "value": 0.97, "method": "cramers_v"},
        ]
        result = _correlated_partners("a", pairs, 0.9)
        assert len(result) == 1
        assert result[0]["value"] == 0.97

    def test_sorted_descending(self):
        pairs = [
            {"var1": "a", "var2": "b", "value": 0.92, "method": "pearson"},
            {"var1": "a", "var2": "c", "value": 0.98, "method": "pearson"},
        ]
        result = _correlated_partners("a", pairs, 0.9)
        assert result[0]["value"] > result[1]["value"]

    def test_unrelated_pairs_ignored(self):
        pairs = [{"var1": "x", "var2": "y", "value": 0.99, "method": "pearson"}]
        result = _correlated_partners("a", pairs, 0.9)
        assert result == []


# ---------------------------------------------------------------------------
# _role
# ---------------------------------------------------------------------------

class TestRole:
    def _col(self, sequence=None, taxonomy=None, measurement_data=None):
        c = MagicMock()
        c.sequence = sequence
        c.taxonomy = taxonomy
        c.measurement_data = measurement_data
        return c

    def test_empty(self):
        assert _role(self._col(), is_empty=True) == "empty"

    def test_dna(self):
        assert _role(self._col(sequence="dna"), is_empty=False) == "dna_rna_sequence"

    def test_protein(self):
        assert _role(self._col(sequence="protein"), is_empty=False) == "protein_sequence"

    def test_taxonomy(self):
        tax = MagicMock(); tax.is_taxonomy = True
        assert _role(self._col(taxonomy=tax), is_empty=False) == "taxonomy"

    def test_measurement(self):
        m = MagicMock()
        assert _role(self._col(measurement_data=m), is_empty=False) == "measurement"

    def test_none(self):
        assert _role(self._col(), is_empty=False) is None


# ---------------------------------------------------------------------------
# _column_entry
# ---------------------------------------------------------------------------

class TestColumnEntry:
    def _base_col(self, name="col", col_type="object", sequence=None):
        c = MagicMock()
        c.name = name
        c.type = col_type
        c.sequence = sequence
        c.number = 100; c.unique = 50; c.missing = 5
        c.missing_per = 5.0; c.density = 0.95; c.constant = False
        c.mixed_types = []; c.suspect_values = []; c.monotonicity = None
        c.cardinality_dimension_ratio = 0.5; c.invalid_seqs = []
        c.taxonomy = None; c.measurement_data = None
        c.protein_data = None; c.dna_rna_data = None
        return c

    def test_empty_column(self):
        c = self._base_col()
        mv = MagicMock(); mv.feature_target_correlation = None; mv.mutual_information = None; mv.top_associations = None
        result = _column_entry(c, None, None, mv, is_empty=True)
        assert result["empty"] is True
        assert "general" not in result

    def test_regular_column_no_blocks(self):
        c = self._base_col()
        mv = MagicMock(); mv.feature_target_correlation = None; mv.mutual_information = None; mv.top_associations = None
        result = _column_entry(c, None, None, mv, is_empty=False)
        assert result["name"] == "col"
        assert "general" in result
        assert "numeric" not in result
        assert "categorical" not in result

    def test_numeric_block_included(self):
        c = self._base_col()
        n = MagicMock()
        n.min = 0.0; n.max = 1.0; n.mean = 0.5; n.median = 0.5; n.mode = 0.5
        n.std = 0.1; n.sum = 50.0; n.kurtosis = 0.0; n.skewness = 0.0
        n.coefficient_of_variation = 0.2; n.mad = 0.05; n.quantiles = []
        n.infinity = 0; n.negative_count = 0; n.zero_count = 0; n.outliers = None
        mv = MagicMock(); mv.feature_target_correlation = None; mv.mutual_information = None; mv.top_associations = None
        result = _column_entry(c, n, None, mv, is_empty=False)
        assert "numeric" in result

    def test_target_relation_included(self):
        c = self._base_col(name="feat")
        mv = MagicMock()
        mv.feature_target_correlation = {"feat": {"value": 0.75, "method": "pearson"}}
        mv.mutual_information = {"feat": {"value": 0.3}}
        mv.top_associations = None
        result = _column_entry(c, None, None, mv, is_empty=False)
        assert "target_relation" in result
        assert result["target_relation"]["association"] == 0.75

    def test_dna_sequence_block(self):
        c = self._base_col(sequence="dna")
        dna = MagicMock()
        for attr in ["gc_content", "ambiguous_base_ratio", "length_stats",
                     "codon_completeness", "gc_skew", "at_skew",
                     "cpg_observed_expected", "tpa_observed_expected", "low_complexity"]:
            s = MagicMock(); s.min = 0.0; s.max = 1.0; s.mean = 0.5
            setattr(dna, attr, s)
        dna.length_outliers = None; dna.reverse_complement_ratio = 0.0
        c.dna_rna_data = dna
        mv = MagicMock(); mv.feature_target_correlation = None; mv.mutual_information = None; mv.top_associations = None
        result = _column_entry(c, None, None, mv, is_empty=False)
        assert "sequence" in result

    def test_missing_per_none(self):
        """missing_per=None should default to 0.0 without crashing."""
        c = self._base_col()
        c.missing_per = None
        mv = MagicMock(); mv.feature_target_correlation = None; mv.mutual_information = None; mv.top_associations = None
        result = _column_entry(c, None, None, mv, is_empty=False)
        assert result["general"]["missing_rate"] == 0.0


# ---------------------------------------------------------------------------
# _quality_block
# ---------------------------------------------------------------------------

class TestQualityBlock:
    def test_none_returns_none(self):
        assert _quality_block(None) is None

    def test_quality_block_structure(self):
        q = MagicMock()
        q.overall = "PASS"; q.passed = 15; q.warnings = 1; q.failed = 0; q.total = 16
        cat = MagicMock(); cat.name = "Dataset"; cat.status = "pass"
        check = MagicMock(); check.name = "Duplicates"; check.status = "pass"; check.message = "None found"
        cat.checks = [check]
        q.categories = [cat]
        result = _quality_block(q)
        assert result["overall"] == "PASS"
        assert result["categories"][0]["name"] == "Dataset"
        assert result["categories"][0]["checks"][0]["message"] == "None found"


# ---------------------------------------------------------------------------
# build_result
# ---------------------------------------------------------------------------

class TestBuildResult:
    def _general(self):
        g = MagicMock()
        g.filename = "test.csv"; g.rows = 100; g.cols = 5
        g.nulls = 2; g.nulls_percentage = 0.02; g.empty_rows = 0
        g.dup_row = 0; g.dup_col = 0; g.ratio = 20.0; g.memory = 1024
        g.n_number = 3; g.n_dna = 0; g.n_rna = 0; g.n_protein = 0
        g.n_taxonomy = 0; g.n_unit = 0; g.n_functional = 0
        g.n_categorical = 2; g.n_empty = 0
        return g

    def _mv(self):
        mv = MagicMock()
        mv.target_name = None
        mv.feature_target_correlation = None
        mv.mutual_information = None
        mv.top_associations = None
        return mv

    def test_meta_structure(self):
        g = self._general()
        result = build_result(g, [], [], [], self._mv(), None, [])
        assert result["meta"]["schema_version"] == "1.0"
        assert result["meta"]["correlation_threshold"] == CORR_THRESHOLD
        assert "generated_at" in result["meta"]

    def test_summary_fields(self):
        g = self._general()
        result = build_result(g, [], [], [], self._mv(), None, [])
        assert result["summary"]["filename"] == "test.csv"
        assert result["summary"]["rows"] == 100

    def test_parameters_passed(self):
        g = self._general()
        params = {"tax": False, "kmer": 3}
        result = build_result(g, [], [], [], self._mv(), None, [], parameters=params)
        assert result["meta"]["parameters"] == params

    def test_columns_list(self):
        g = self._general()
        col = MagicMock()
        col.name = "a"; col.type = "int64"; col.sequence = None
        col.number = 10; col.unique = 5; col.missing = 0
        col.missing_per = 0.0; col.density = 1.0; col.constant = False
        col.mixed_types = []; col.suspect_values = []; col.monotonicity = None
        col.cardinality_dimension_ratio = 0.5; col.invalid_seqs = []
        col.taxonomy = None; col.measurement_data = None
        col.protein_data = None; col.dna_rna_data = None
        result = build_result(g, [col], [], [], self._mv(), None, [])
        assert len(result["columns"]) == 1

    def test_empty_cols_marked(self):
        g = self._general()
        col = MagicMock()
        col.name = "empty_col"; col.type = "float64"; col.sequence = None
        col.taxonomy = None; col.measurement_data = None
        result = build_result(g, [col], [], [], self._mv(), None, ["empty_col"])
        assert result["columns"][0].get("empty") is True


# ---------------------------------------------------------------------------
# write_result_json
# ---------------------------------------------------------------------------

class TestWriteResultJson:
    def _general(self):
        g = MagicMock()
        g.filename = "t.csv"; g.rows = 10; g.cols = 2
        g.nulls = 0; g.nulls_percentage = 0.0; g.empty_rows = 0
        g.dup_row = 0; g.dup_col = 0; g.ratio = 5.0; g.memory = 512
        g.n_number = 1; g.n_dna = 0; g.n_rna = 0; g.n_protein = 0
        g.n_taxonomy = 0; g.n_unit = 0; g.n_functional = 0
        g.n_categorical = 1; g.n_empty = 0
        return g

    def _mv(self):
        mv = MagicMock()
        mv.target_name = None
        mv.feature_target_correlation = None
        mv.mutual_information = None
        mv.top_associations = None
        return mv

    def test_creates_valid_json_file(self, tmp_path):
        path = tmp_path / "results.json"
        write_result_json(path, self._general(), [], [], [], self._mv(), None, [])
        assert path.exists()
        data = json.loads(path.read_text(encoding="utf-8"))
        assert "meta" in data
        assert "summary" in data
        assert "columns" in data

    def test_returns_path(self, tmp_path):
        path = tmp_path / "results.json"
        result = write_result_json(path, self._general(), [], [], [], self._mv(), None, [])
        assert result == path

    def test_file_is_utf8(self, tmp_path):
        path = tmp_path / "results.json"
        write_result_json(path, self._general(), [], [], [], self._mv(), None, [])
        path.read_text(encoding="utf-8")  # must not raise


# ---------------------------------------------------------------------------
# _json_default
# ---------------------------------------------------------------------------

class TestJsonDefault:
    def test_np_integer(self):
        assert _json_default(np.int64(5)) == 5

    def test_np_floating_finite(self):
        assert _json_default(np.float64(1.5)) == 1.5

    def test_np_floating_nan_returns_none(self):
        assert _json_default(np.float64(float("nan"))) is None

    def test_np_ndarray(self):
        assert _json_default(np.array([1, 2])) == [1, 2]

    def test_has_tolist(self):
        obj = MagicMock()
        obj.tolist.return_value = [1, 2]
        del obj.item  # ensure item path not taken
        # tolist path
        import pandas as pd
        s = pd.Series([1, 2])
        assert _json_default(s) == [1, 2]

    def test_unknown_type_stringified(self):
        class Custom:
            def __str__(self): return "custom"
        result = _json_default(Custom())
        assert result == "custom"