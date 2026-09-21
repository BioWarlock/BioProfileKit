import json
import math
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from cli.report_json import (
    _num,
    _present,
    _listify,
    _summary,
    _outliers,
    _numeric_block,
    _categorical_block,
    _dna_block,
    _protein_block,
    _taxonomy_block,
    _measurement_block,
    _correlated_partners,
    _role,
    _column_entry,
    _quality_block,
    build_result,
    _json_default,
    write_result_json,
    SCHEMA_VERSION,
    CORR_THRESHOLD,
)


# ---------------------------------------------------------------------------
# _num
# ---------------------------------------------------------------------------

class TestNum:
    def test_none_stays_none(self):
        assert _num(None) is None

    def test_numpy_integer_converted_to_int(self):
        result = _num(np.int64(42))
        assert result == 42
        assert isinstance(result, int)

    def test_numpy_float_converted_to_float(self):
        result = _num(np.float64(3.5))
        assert result == 3.5
        assert isinstance(result, float)

    def test_plain_float_passes_through(self):
        assert _num(3.14) == 3.14

    def test_nan_becomes_none(self):
        assert _num(np.nan) is None
        assert _num(float("nan")) is None

    def test_infinity_becomes_none(self):
        assert _num(np.inf) is None
        assert _num(-np.inf) is None

    def test_int_bool_str_pass_through_unchanged(self):
        assert _num(5) == 5
        assert _num(True) is True
        assert _num("hello") == "hello"

    def test_numpy_array_converted_to_list(self):
        result = _num(np.array([1.0, 2.0, np.nan]))
        assert result == [1.0, 2.0, None]

    def test_unrecognized_type_returned_as_is(self):
        obj = object()
        assert _num(obj) is obj


# ---------------------------------------------------------------------------
# _present
# ---------------------------------------------------------------------------

class TestPresent:
    def test_none_is_not_present(self):
        assert _present(None) is False

    def test_empty_array_not_present(self):
        assert _present(np.array([])) is False

    def test_nonempty_array_present(self):
        assert _present(np.array([1, 2])) is True

    def test_empty_series_not_present(self):
        assert _present(pd.Series([], dtype=float)) is False

    def test_nonempty_series_present(self):
        assert _present(pd.Series([1, 2])) is True

    def test_empty_collections_not_present(self):
        assert _present([]) is False
        assert _present(()) is False
        assert _present(set()) is False
        assert _present({}) is False
        assert _present("") is False

    def test_nonempty_collections_present(self):
        assert _present([1]) is True
        assert _present("x") is True

    def test_falls_back_to_bool_for_scalars(self):
        assert _present(0) is False
        assert _present(5) is True


# ---------------------------------------------------------------------------
# _listify
# ---------------------------------------------------------------------------

class TestListify:
    def test_not_present_returns_none(self):
        assert _listify(None) is None
        assert _listify([]) is None

    def test_numpy_array_listified_and_cleaned(self):
        result = _listify(np.array([1.0, np.nan, 3.0]))
        assert result == [1.0, None, 3.0]

    def test_pandas_series_listified(self):
        result = _listify(pd.Series([1, 2, 3]))
        assert result == [1, 2, 3]

    def test_list_tuple_set_converted(self):
        assert _listify([1, 2]) == [1, 2]
        assert _listify((1, 2)) == [1, 2]
        assert sorted(_listify({1, 2})) == [1, 2]

    def test_scalar_falls_back_to_num(self):
        assert _listify(5) == 5
        assert _listify("x") == "x"


# ---------------------------------------------------------------------------
# _summary / _outliers
# ---------------------------------------------------------------------------

class TestSummary:
    def test_none_returns_none(self):
        assert _summary(None) is None

    def test_extracts_min_max_mean(self):
        s = SimpleNamespace(min=1.0, max=9.0, mean=5.0)
        assert _summary(s) == {"min": 1.0, "max": 9.0, "mean": 5.0}


class TestOutliers:
    def test_none_returns_none(self):
        assert _outliers(None) is None

    def test_extracts_all_fields(self):
        o = SimpleNamespace(
            lower_bound=1.0, upper_bound=9.0, medcouple=0.1,
            n_lower_iqr=1, n_upper_iqr=2, n_lower_mzscore=0,
            n_upper_mzscore=1, n_lower_zscore=0, n_upper_zscore=1,
        )
        result = _outliers(o)
        assert result["lower_bound"] == 1.0
        assert result["n_upper_iqr"] == 2


# ---------------------------------------------------------------------------
# _numeric_block / _categorical_block
# ---------------------------------------------------------------------------

def make_numeric(**overrides):
    defaults = dict(
        min=1.0, max=10.0, mean=5.5, median=5.0, mode=5.0, std=2.0,
        sum=55.0, kurtosis=0.1, skewness=0.2, coefficient_of_variation=0.36,
        mad=1.5, quantiles=np.array([2.0, 5.0, 8.0]), infinity=0,
        negative_count=0, zero_count=0, outliers=None,
    )
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


class TestNumericBlock:
    def test_none_returns_none(self):
        assert _numeric_block(None) is None

    def test_full_block(self):
        block = _numeric_block(make_numeric())
        assert block["mean"] == 5.5
        assert block["quantiles"] == [2.0, 5.0, 8.0]
        assert block["outliers"] is None

    def test_with_outliers(self):
        outliers = SimpleNamespace(
            lower_bound=0.0, upper_bound=10.0, medcouple=0.0,
            n_lower_iqr=1, n_upper_iqr=0, n_lower_mzscore=1,
            n_upper_mzscore=0, n_lower_zscore=1, n_upper_zscore=0,
        )
        block = _numeric_block(make_numeric(outliers=outliers))
        assert block["outliers"]["n_lower_iqr"] == 1


def make_categorical(**overrides):
    defaults = dict(
        unique_categories=3, mode="A", entropy=1.1, gini=0.5,
        simpson_diversity=0.6, max_category_length=5, min_category_length=1,
        cardinality_ratio=0.3, rare_categories=1, top_5_coverage=0.9,
        top_1_coverage=0.4, cib_ratio=0.2, effective_cardinality=2.5,
    )
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


class TestCategoricalBlock:
    def test_none_returns_none(self):
        assert _categorical_block(None) is None

    def test_full_block(self):
        block = _categorical_block(make_categorical())
        assert block["mode"] == "A"
        assert block["unique_categories"] == 3


# ---------------------------------------------------------------------------
# _dna_block / _protein_block
# ---------------------------------------------------------------------------

def make_summary(min=1.0, max=2.0, mean=1.5):
    return SimpleNamespace(min=min, max=max, mean=mean)


class TestDnaBlock:
    def test_full_block(self):
        d = SimpleNamespace(
            gc_content=make_summary(), ambiguous_base_ratio=make_summary(),
            length_stats=make_summary(), length_outliers=None,
            codon_completeness=make_summary(), gc_skew=make_summary(),
            at_skew=make_summary(), cpg_observed_expected=make_summary(),
            tpa_observed_expected=make_summary(), low_complexity=make_summary(),
            reverse_complement_ratio=3.2,
        )
        block = _dna_block(d)
        assert block["gc_content"] == {"min": 1.0, "max": 2.0, "mean": 1.5}
        assert block["reverse_complement_ratio"] == 3.2
        assert block["length_outliers"] is None


class TestProteinBlock:
    def test_full_block(self):
        p = SimpleNamespace(
            ambiguous_residue_ratio=make_summary(), length_stats=make_summary(),
            length_outliers=None, stop_codon_ratio=1.5, low_complexity=make_summary(),
            gravy=make_summary(), cysteine_count=make_summary(),
            disorder_propensity=make_summary(),
            aa_group_distribution={"Polar": 0.3, "Unpolar": 0.7},
        )
        block = _protein_block(p)
        assert block["stop_codon_ratio"] == 1.5
        assert block["aa_group_distribution"] == {"Polar": 0.3, "Unpolar": 0.7}

    def test_none_aa_group_distribution_becomes_none(self):
        p = SimpleNamespace(
            ambiguous_residue_ratio=make_summary(), length_stats=make_summary(),
            length_outliers=None, stop_codon_ratio=0.0, low_complexity=make_summary(),
            gravy=make_summary(), cysteine_count=make_summary(),
            disorder_propensity=make_summary(), aa_group_distribution=None,
        )
        block = _protein_block(p)
        assert block["aa_group_distribution"] is None

    def test_empty_dict_aa_group_distribution_becomes_none(self):
        """An empty dict is falsy, so it should also collapse to None
        (same branch as the explicit None case)."""
        p = SimpleNamespace(
            ambiguous_residue_ratio=make_summary(), length_stats=make_summary(),
            length_outliers=None, stop_codon_ratio=0.0, low_complexity=make_summary(),
            gravy=make_summary(), cysteine_count=make_summary(),
            disorder_propensity=make_summary(), aa_group_distribution={},
        )
        block = _protein_block(p)
        assert block["aa_group_distribution"] is None


# ---------------------------------------------------------------------------
# _taxonomy_block
# ---------------------------------------------------------------------------

class TestTaxonomyBlock:
    def test_none_returns_none(self):
        assert _taxonomy_block(None) is None

    def test_not_taxonomy_returns_none(self):
        tax = SimpleNamespace(is_taxonomy=False)
        assert _taxonomy_block(tax) is None

    def test_minimal_valid_taxonomy(self):
        tax = SimpleNamespace(is_taxonomy=True, is_mixed=False)
        block = _taxonomy_block(tax)
        assert block == {"is_taxonomy": True, "is_mixed": False}

    def test_rank_distribution_included_when_present(self):
        tax = SimpleNamespace(is_taxonomy=True, is_mixed=True, rank_distribution={"species": 1.0})
        block = _taxonomy_block(tax)
        assert block["rank_distribution"] == {"species": 1.0}
        assert block["is_mixed"] is True

    def test_outdated_names_included_when_present(self):
        tax = SimpleNamespace(is_taxonomy=True, is_mixed=False, outdated_names={"Old": "New"})
        block = _taxonomy_block(tax)
        assert block["name_corrections"] == {"Old": "New"}

    def test_invalid_names_included_when_present(self):
        tax = SimpleNamespace(is_taxonomy=True, is_mixed=False, invalid_names=["Bad Name"])
        block = _taxonomy_block(tax)
        assert block["invalid_values"] == ["Bad Name"]

    def test_falsy_optional_fields_omitted(self):
        tax = SimpleNamespace(
            is_taxonomy=True, is_mixed=False,
            rank_distribution=None, outdated_names=None, invalid_names=None,
        )
        block = _taxonomy_block(tax)
        assert "rank_distribution" not in block
        assert "name_corrections" not in block
        assert "invalid_values" not in block


# ---------------------------------------------------------------------------
# _measurement_block
# ---------------------------------------------------------------------------

class TestMeasurementBlock:
    def test_none_returns_none(self):
        assert _measurement_block(None) is None

    def test_full_block(self):
        m = SimpleNamespace(units=["mg"], unit_counts={"mg": 5}, with_measurement=True)
        block = _measurement_block(m)
        assert block == {"units": ["mg"], "unit_counts": {"mg": 5}, "with_measurement": True}

    def test_unit_counts_keys_stringified(self):
        m = SimpleNamespace(units=["mg"], unit_counts={5: 2}, with_measurement=False)
        block = _measurement_block(m)
        assert block["unit_counts"] == {"5": 2}


# ---------------------------------------------------------------------------
# _correlated_partners
# ---------------------------------------------------------------------------

class TestCorrelatedPartners:
    def test_empty_associations_returns_empty_list(self):
        assert _correlated_partners("col_a", [], 0.9) == []
        assert _correlated_partners("col_a", None, 0.9) == []

    def test_below_threshold_excluded(self):
        assocs = [{"var1": "col_a", "var2": "col_b", "value": 0.5, "method": "pearson"}]
        assert _correlated_partners("col_a", assocs, 0.9) == []

    def test_column_not_involved_excluded(self):
        assocs = [{"var1": "col_x", "var2": "col_y", "value": 0.95, "method": "pearson"}]
        assert _correlated_partners("col_a", assocs, 0.9) == []

    def test_partner_identified_regardless_of_position(self):
        assocs = [
            {"var1": "col_a", "var2": "col_b", "value": 0.95, "method": "pearson"},
            {"var1": "col_c", "var2": "col_a", "value": 0.92, "method": "spearman"},
        ]
        result = _correlated_partners("col_a", assocs, 0.9)
        partners = {r["column"] for r in result}
        assert partners == {"col_b", "col_c"}

    def test_duplicate_partner_keeps_max_value(self):
        assocs = [
            {"var1": "col_a", "var2": "col_b", "value": 0.91, "method": "pearson"},
            {"var1": "col_a", "var2": "col_b", "value": 0.98, "method": "spearman"},
        ]
        result = _correlated_partners("col_a", assocs, 0.9)
        assert len(result) == 1
        assert result[0]["value"] == 0.98
        assert result[0]["method"] == "spearman"

    def test_results_sorted_descending_by_value(self):
        assocs = [
            {"var1": "col_a", "var2": "col_b", "value": 0.91, "method": "pearson"},
            {"var1": "col_a", "var2": "col_c", "value": 0.99, "method": "pearson"},
        ]
        result = _correlated_partners("col_a", assocs, 0.9)
        assert [r["column"] for r in result] == ["col_c", "col_b"]


# ---------------------------------------------------------------------------
# _role
# ---------------------------------------------------------------------------

class TestRole:
    def test_empty_column_role(self):
        col = SimpleNamespace(sequence=None)
        assert _role(col, is_empty=True) == "empty"

    def test_dna_sequence_role(self):
        col = SimpleNamespace(sequence="dna")
        assert _role(col, is_empty=False) == "dna_rna_sequence"

    def test_protein_sequence_role(self):
        col = SimpleNamespace(sequence="protein")
        assert _role(col, is_empty=False) == "protein_sequence"

    def test_taxonomy_role(self):
        col = SimpleNamespace(sequence=None, taxonomy=SimpleNamespace(is_taxonomy=True))
        assert _role(col, is_empty=False) == "taxonomy"

    def test_measurement_role(self):
        col = SimpleNamespace(sequence=None, taxonomy=None, measurement_data=SimpleNamespace())
        assert _role(col, is_empty=False) == "measurement"

    def test_no_role_returns_none(self):
        col = SimpleNamespace(sequence=None, taxonomy=None, measurement_data=None)
        assert _role(col, is_empty=False) is None

    def test_taxonomy_not_is_taxonomy_falls_through(self):
        col = SimpleNamespace(
            sequence=None, taxonomy=SimpleNamespace(is_taxonomy=False), measurement_data=None,
        )
        assert _role(col, is_empty=False) is None


# ---------------------------------------------------------------------------
# _column_entry
# ---------------------------------------------------------------------------

def make_column_overview(**overrides):
    defaults = dict(
        name="col1", type="float64", number=10, unique=8, missing=2,
        missing_per=20.0, density=0.8, constant=False, mixed_types=[],
        suspect_values=[], monotonicity=None, cardinality_dimension_ratio=0.5,
        invalid_seqs=[], sequence=None, protein_data=None, dna_rna_data=None,
        taxonomy=None, measurement_data=None,
    )
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


class TestColumnEntry:
    def test_empty_column_short_circuits(self):
        col = make_column_overview()
        entry = _column_entry(col, None, None, SimpleNamespace(), is_empty=True)
        assert entry == {"name": "col1", "dtype": "float64", "role": "empty", "empty": True}

    def test_basic_numeric_column(self):
        col = make_column_overview()
        num = make_numeric()
        entry = _column_entry(col, num, None, SimpleNamespace(), is_empty=False)
        assert entry["general"]["count"] == 10
        assert entry["general"]["missing_rate"] == 0.2
        assert entry["numeric"]["mean"] == 5.5
        assert "categorical" not in entry

    def test_categorical_column(self):
        col = make_column_overview(type="object")
        cat = make_categorical()
        entry = _column_entry(col, None, cat, SimpleNamespace(), is_empty=False)
        assert entry["categorical"]["mode"] == "A"
        assert "numeric" not in entry

    def test_dna_sequence_column(self):
        col = make_column_overview(
            sequence="dna",
            dna_rna_data=SimpleNamespace(
                gc_content=make_summary(), ambiguous_base_ratio=make_summary(),
                length_stats=make_summary(), length_outliers=None,
                codon_completeness=make_summary(), gc_skew=make_summary(),
                at_skew=make_summary(), cpg_observed_expected=make_summary(),
                tpa_observed_expected=make_summary(), low_complexity=make_summary(),
                reverse_complement_ratio=1.0,
            ),
        )
        entry = _column_entry(col, None, None, SimpleNamespace(), is_empty=False)
        assert entry["role"] == "dna_rna_sequence"
        assert "sequence" in entry
        assert entry["sequence"]["reverse_complement_ratio"] == 1.0

    def test_taxonomy_column(self):
        col = make_column_overview(taxonomy=SimpleNamespace(is_taxonomy=True, is_mixed=False))
        entry = _column_entry(col, None, None, SimpleNamespace(), is_empty=False)
        assert entry["role"] == "taxonomy"
        assert entry["taxonomy"] == {"is_taxonomy": True, "is_mixed": False}

    def test_measurement_column(self):
        col = make_column_overview(
            measurement_data=SimpleNamespace(units=["mg"], unit_counts={"mg": 3}, with_measurement=True)
        )
        entry = _column_entry(col, None, None, SimpleNamespace(), is_empty=False)
        assert entry["role"] == "measurement"
        assert entry["measurement"]["units"] == ["mg"]

    def test_target_relation_included(self):
        col = make_column_overview(name="col1")
        multivariate = SimpleNamespace(
            feature_target_correlation={"col1": {"value": 0.8, "method": "pearson"}},
            mutual_information={"col1": {"value": 0.4}},
            top_associations=None,
        )
        entry = _column_entry(col, None, None, multivariate, is_empty=False)
        assert entry["target_relation"]["association"] == 0.8
        assert entry["target_relation"]["mutual_information"] == 0.4

    def test_no_target_relation_when_absent(self):
        col = make_column_overview(name="col1")
        multivariate = SimpleNamespace(feature_target_correlation=None, mutual_information=None, top_associations=None)
        entry = _column_entry(col, None, None, multivariate, is_empty=False)
        assert "target_relation" not in entry

    def test_correlated_with_included(self):
        col = make_column_overview(name="col1")
        multivariate = SimpleNamespace(
            feature_target_correlation=None, mutual_information=None,
            top_associations=[{"var1": "col1", "var2": "col2", "value": 0.95, "method": "pearson"}],
        )
        entry = _column_entry(col, None, None, multivariate, is_empty=False)
        assert entry["correlated_with"][0]["column"] == "col2"


# ---------------------------------------------------------------------------
# _quality_block
# ---------------------------------------------------------------------------

class TestQualityBlock:
    def test_none_returns_none(self):
        assert _quality_block(None) is None

    def test_full_block(self):
        check = SimpleNamespace(name="Sequence Validity", status="pass", message="All good")
        category = SimpleNamespace(name="Biological", status="pass", checks=[check])
        quality = SimpleNamespace(overall="pass", passed=1, warnings=0, failed=0, total=1, categories=[category])
        block = _quality_block(quality)
        assert block["overall"] == "pass"
        assert block["categories"][0]["checks"][0]["name"] == "Sequence Validity"


# ---------------------------------------------------------------------------
# build_result / write_result_json — integration
# ---------------------------------------------------------------------------

def make_general(**overrides):
    defaults = dict(
        filename="test.csv", rows=100, cols=3, nulls=5, nulls_percentage=1.5,
        empty_rows=0, dup_row=0, dup_col=0, ratio=33.3, memory=2048,
        n_number=1, n_dna=0, n_rna=0, n_protein=0, n_taxonomy=1, n_unit=0,
        n_functional=0, n_categorical=1, n_empty=0,
    )
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


class TestBuildResult:
    def test_meta_and_summary_present(self):
        general = make_general()
        result = build_result(general, [], [], [], SimpleNamespace(), None, [])
        assert result["meta"]["schema_version"] == SCHEMA_VERSION
        assert result["meta"]["correlation_threshold"] == CORR_THRESHOLD
        assert result["summary"]["filename"] == "test.csv"
        assert result["summary"]["rows"] == 100
        assert result["columns"] == []

    def test_parameters_default_to_empty_dict(self):
        general = make_general()
        result = build_result(general, [], [], [], SimpleNamespace(), None, [])
        assert result["meta"]["parameters"] == {}

    def test_parameters_passed_through(self):
        general = make_general()
        result = build_result(general, [], [], [], SimpleNamespace(), None, [], parameters={"foo": "bar"})
        assert result["meta"]["parameters"] == {"foo": "bar"}

    def test_target_name_from_multivariate(self):
        general = make_general()
        multivariate = SimpleNamespace(target_name="col1")
        result = build_result(general, [], [], [], multivariate, None, [])
        assert result["target"] == "col1"

    def test_columns_assembled_with_matching_overviews(self):
        general = make_general()
        col = make_column_overview(name="col1")
        num = make_numeric()
        num.name = "col1"
        result = build_result(general, [col], [num], [], SimpleNamespace(), None, [])
        assert len(result["columns"]) == 1
        assert result["columns"][0]["numeric"]["mean"] == 5.5

    def test_empty_columns_marked_from_empty_cols_list(self):
        general = make_general()
        col = make_column_overview(name="col1")
        result = build_result(general, [col], [], [], SimpleNamespace(), None, ["col1"])
        assert result["columns"][0]["empty"] is True


class TestWriteResultJson:
    def test_writes_valid_json_file(self, tmp_path):
        general = make_general()
        out_path = tmp_path / "result.json"
        write_result_json(str(out_path), general, [], [], [], SimpleNamespace(), None, [])
        assert out_path.is_file()
        with open(out_path, encoding="utf-8") as f:
            data = json.load(f)
        assert data["summary"]["filename"] == "test.csv"

    def test_returns_the_path(self, tmp_path):
        general = make_general()
        out_path = tmp_path / "result.json"
        returned = write_result_json(str(out_path), general, [], [], [], SimpleNamespace(), None, [])
        assert returned == str(out_path)

    def test_json_default_handles_leftover_numpy_values(self, tmp_path):
        """Even if a raw numpy value slipped past _num/_listify cleaning
        somewhere, json.dump must not crash thanks to the default=... hook."""
        general = make_general()
        col = make_column_overview(name="col1", mixed_types=[np.int64(3)])
        out_path = tmp_path / "result.json"
        write_result_json(str(out_path), general, [col], [], [], SimpleNamespace(), None, [])
        with open(out_path, encoding="utf-8") as f:
            data = json.load(f)
        assert data["columns"][0]["general"]["mixed_types"] == [3]


# ---------------------------------------------------------------------------
# _json_default
# ---------------------------------------------------------------------------

class TestJsonDefault:
    def test_numpy_integer(self):
        assert _json_default(np.int32(7)) == 7

    def test_numpy_finite_float(self):
        assert _json_default(np.float32(1.5)) == pytest.approx(1.5)

    def test_numpy_nan_becomes_none(self):
        assert _json_default(np.float64(np.nan)) is None

    def test_numpy_array(self):
        assert _json_default(np.array([1, 2, 3])) == [1, 2, 3]

    def test_object_with_tolist(self):
        assert _json_default(pd.Series([1, 2])) == [1, 2]

    def test_object_with_item(self):
        assert _json_default(np.int64(9)) == 9

    def test_fallback_to_str(self):
        class Custom:
            def __str__(self):
                return "custom-repr"
        assert _json_default(Custom()) == "custom-repr"