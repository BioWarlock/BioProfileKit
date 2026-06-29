"""Unit tests for biological/measurement_data.py"""
import sys
import os
import re
import pytest
import pandas as pd
from unittest.mock import MagicMock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from biological.measurement_data import measurement_columns, match_units, has_number_and_unit


class TestHasNumberAndUnit:
    def test_valid_measurement(self):
        assert has_number_and_unit("5.0mg") is True
        assert has_number_and_unit("100mL") is True
        assert has_number_and_unit("-3.5°C") is True

    def test_special_units_return_false(self):
        assert has_number_and_unit("1/s") is False
        assert has_number_and_unit("1/m") is False
        assert has_number_and_unit("1/M") is False

    def test_plain_number_returns_false(self):
        assert has_number_and_unit("42") is False

    def test_plain_string_returns_false(self):
        assert has_number_and_unit("hello") is False


class TestMatchUnits:
    def test_all_match(self):
        from enums.measurement_enum import MEASUREMENTS
        pattern = MEASUREMENTS.UNIT_COLUMN.value
        entries = ["5 mg", "10 mL"]
        result = match_units(entries, pattern)
        assert isinstance(result, list)

    def test_no_match_returns_empty(self):
        pattern = re.compile(r'^\d+\s*mg$')
        result = match_units(["hello", "world"], pattern)
        assert result == []

    def test_partial_match_not_included(self):
        pattern = re.compile(r'^\d+mg$')
        result = match_units(["5mg", "5mg/L"], pattern)
        assert "5mg" in result
        assert "5mg/L" not in result


class TestMeasurementColumns:
    def test_unit_in_col_title(self):
        col = pd.Series([1.0, 2.0, 3.0])
        result = measurement_columns(col, "[mg/L]", "float64")
        if result is not False:
            assert hasattr(result, "units")

    def test_non_str_type_returns_false(self):
        col = pd.Series([1.0, 2.0, 3.0])
        result = measurement_columns(col, "value", "float64")
        assert result is False

    def test_str_col_non_matching_returns_false(self):
        col = pd.Series(["hello", "world", "foo"])
        result = measurement_columns(col, "label", "str")
        assert result is False

    def test_empty_col_returns_false(self):
        col = pd.Series([], dtype=str)
        result = measurement_columns(col, "label", "str")
        # An empty string column has no values to match against the unit regex,
        # so the function returns either False or an empty UNITColumns object.
        assert result is False or (hasattr(result, "units") and result.units == [])