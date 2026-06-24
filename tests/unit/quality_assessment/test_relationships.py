"""Unit tests for quality_assessment/relationships.py"""
import sys
import os
import pytest
from unittest.mock import MagicMock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from quality_assessment.relationships import _check_leakage, _check_multicollinearity


class TestCheckLeakage:
    def test_no_target_passes(self):
        mv = MagicMock(); mv.feature_target_correlation = None
        result = _check_leakage(mv)
        assert result.status == "pass"
        assert "skipped" in result.message

    def test_no_suspects_passes(self):
        mv = MagicMock()
        mv.feature_target_correlation = {"a": {"value": 0.5}, "b": {"value": 0.3}}
        result = _check_leakage(mv)
        assert result.status == "pass"

    def test_warn_level(self):
        mv = MagicMock()
        mv.feature_target_correlation = {"feat": {"value": 0.93}}
        result = _check_leakage(mv)
        assert result.status == "warn"
        assert "feat" in result.message

    def test_fail_level(self):
        mv = MagicMock()
        mv.feature_target_correlation = {"feat": {"value": 0.99}}
        result = _check_leakage(mv)
        assert result.status == "fail"

    def test_multiple_suspects_sorted(self):
        mv = MagicMock()
        mv.feature_target_correlation = {
            "a": {"value": 0.91},
            "b": {"value": 0.95},
        }
        result = _check_leakage(mv)
        assert "b" in result.message


class TestCheckMulticollinearity:
    def test_no_pairs_passes(self):
        mv = MagicMock(); mv.top_associations = None
        result = _check_multicollinearity(mv)
        assert result.status == "pass"

    def test_all_below_threshold_passes(self):
        mv = MagicMock()
        mv.top_associations = [{"var1": "a", "var2": "b", "value": 0.5}]
        result = _check_multicollinearity(mv)
        assert result.status == "pass"

    def test_above_threshold_warns(self):
        mv = MagicMock()
        mv.top_associations = [{"var1": "a", "var2": "b", "value": 0.95}]
        result = _check_multicollinearity(mv)
        assert result.status == "warn"
        assert "a↔b" in result.message

    def test_empty_list_passes(self):
        mv = MagicMock(); mv.top_associations = []
        result = _check_multicollinearity(mv)
        assert result.status == "pass"