import re
from types import SimpleNamespace
from unittest.mock import patch

import pandas as pd
import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from biological.measurement_data import (
    measurement_columns,
    match_units,
    has_number_and_unit,
)

MEASUREMENTS_PATCH = "biological.measurement_data.MEASUREMENTS"


def fake_measurements(title_pattern=r"(?!)x", column_pattern=r"-?\d+\.?\d*\s*[a-zA-Z°/%]+"):
    return SimpleNamespace(
        UNIT_IN_COL_TITLE=SimpleNamespace(value=re.compile(title_pattern)),
        UNIT_COLUMN=SimpleNamespace(value=re.compile(column_pattern)),
    )


# ---------------------------------------------------------------------------
# match_units
# ---------------------------------------------------------------------------

class TestMatchUnits:
    def test_matches_full_pattern_entries(self):
        pattern = re.compile(r"-?\d+\.?\d*\s*[a-zA-Z]+")
        result = match_units(["5 mg", "not a match!!", "10 kg"], pattern)
        assert result == ["5 mg", "10 kg"]

    def test_no_matches_returns_empty_list(self):
        pattern = re.compile(r"^ONLY_THIS$")
        result = match_units(["foo", "bar"], pattern)
        assert result == []

    def test_empty_entries_returns_empty_list(self):
        pattern = re.compile(r".*")
        assert match_units([], pattern) == []


# ---------------------------------------------------------------------------
# has_number_and_unit
# ---------------------------------------------------------------------------

class TestHasNumberAndUnit:
    def test_number_with_unit_is_true(self):
        assert has_number_and_unit("5 mg") is True
        assert has_number_and_unit("-3.2kg") is True
        assert has_number_and_unit("100%") is True

    def test_unit_only_is_false(self):
        assert has_number_and_unit("mg") is False

    def test_number_only_is_false(self):
        assert has_number_and_unit("42") is False

    def test_special_units_are_false_even_if_pattern_would_match(self):
        for special in ("1/s", "1/m", "1/M", "1/h", "1/min"):
            assert has_number_and_unit(special) is False

    def test_empty_string_is_false(self):
        assert has_number_and_unit("") is False


# ---------------------------------------------------------------------------
# measurement_columns
# ---------------------------------------------------------------------------

class TestMeasurementColumnsTitleMatch:
    def test_unit_extracted_from_bracketed_title(self):
        measurements = fake_measurements(title_pattern=r"\[[a-zA-Z]+\]")
        col = pd.Series([1.0, 2.0, 3.0])
        with patch(MEASUREMENTS_PATCH, measurements):
            result = measurement_columns(col, name="[mg]", col_type="float")
        assert result.units == ["mg"]
        assert result.unit_counts == {"mg": 1}
        assert result.with_measurement is False

    def test_no_title_match_falls_through_to_value_check(self):
        measurements = fake_measurements()  # title pattern matches nothing
        col = pd.Series(["a", "b"])
        with patch(MEASUREMENTS_PATCH, measurements):
            result = measurement_columns(col, name="weight", col_type="float")
        assert result is False


class TestMeasurementColumnsValueMatch:
    def test_all_values_match_unit_pattern(self):
        measurements = fake_measurements(column_pattern=r"-?\d+\.?\d*\s*[a-zA-Z]+")
        col = pd.Series(["5 mg", "10 mg", "15 kg"])
        with patch(MEASUREMENTS_PATCH, measurements):
            result = measurement_columns(col, name="weight", col_type="str")
        assert result is not False
        assert set(result.units) == {"mg", "kg"}
        assert result.with_measurement is True

    def test_mostly_matching_values_pass_via_95pct_threshold(self):
        """96/100 values match -> above the 0.95 fallback threshold."""
        measurements = fake_measurements(column_pattern=r"-?\d+\.?\d*\s*[a-zA-Z]+")
        values = [f"{i} mg" for i in range(96)] + ["junk"] * 4
        col = pd.Series(values)
        with patch(MEASUREMENTS_PATCH, measurements):
            result = measurement_columns(col, name="weight", col_type="str")
        assert result is not False

    def test_too_few_matching_values_returns_false(self):
        measurements = fake_measurements(column_pattern=r"-?\d+\.?\d*\s*[a-zA-Z]+")
        values = [f"{i} mg" for i in range(5)] + ["junk"] * 95
        col = pd.Series(values)
        with patch(MEASUREMENTS_PATCH, measurements):
            result = measurement_columns(col, name="weight", col_type="str")
        assert result is False

    def test_single_char_values_excluded(self):
        """`all(len(x) > 1 for x in values)` guard — single-char unique
        values should never reach the unit-matching step."""
        measurements = fake_measurements(column_pattern=r".*")
        col = pd.Series(["a", "b", "c"])
        with patch(MEASUREMENTS_PATCH, measurements):
            result = measurement_columns(col, name="flag", col_type="str")
        assert result is False

    def test_non_str_col_type_returns_false(self):
        measurements = fake_measurements(column_pattern=r".*")
        col = pd.Series([1, 2, 3])
        with patch(MEASUREMENTS_PATCH, measurements):
            result = measurement_columns(col, name="weight", col_type="int")
        assert result is False

    def test_unit_split_on_space_takes_second_part(self):
        measurements = fake_measurements(column_pattern=r"-?\d+\.?\d*\s*[a-zA-Z]+")
        col = pd.Series(["5 mg", "10 mg"])
        with patch(MEASUREMENTS_PATCH, measurements):
            result = measurement_columns(col, name="weight", col_type="str")
        assert result.units == ["mg", "mg"]

    def test_unit_counts_reflect_raw_value_counts(self):
        measurements = fake_measurements(column_pattern=r"-?\d+\.?\d*\s*[a-zA-Z]+")
        col = pd.Series(["5 mg", "5 mg", "10 kg"])
        with patch(MEASUREMENTS_PATCH, measurements):
            result = measurement_columns(col, name="weight", col_type="str")
        assert result.unit_counts == {"5 mg": 2, "10 kg": 1}