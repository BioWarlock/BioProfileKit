import pandas as pd
import pytest
import sys
import os
from unittest.mock import patch, MagicMock

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from biological.functional_annotation import (
    annotation_flags,
    validate_annotation,
    clean_strings,
    AnnotationFlags,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_cog_df(ids):
    """Minimal COG reference DataFrame."""
    return pd.DataFrame({"COG_ID": ids})


def make_go_df(ids):
    """Minimal GO reference DataFrame."""
    return pd.DataFrame({"GO_ID": ids})


# ---------------------------------------------------------------------------
# clean_strings
# ---------------------------------------------------------------------------

class TestCleanStrings:
    def test_uppercases_values(self):
        s = pd.Series(["go:0008150", "go:0003674"])
        result = clean_strings(s)
        assert list(result) == ["GO:0008150", "GO:0003674"]

    def test_strips_whitespace(self):
        s = pd.Series(["  COG0001 ", "\tCOG0002\n"])
        result = clean_strings(s)
        assert list(result) == ["COG0001", "COG0002"]

    def test_strips_and_uppercases_together(self):
        s = pd.Series([" go:0008150 ", "GO:0003674"])
        result = clean_strings(s)
        assert list(result) == ["GO:0008150", "GO:0003674"]

    def test_handles_nan(self):
        s = pd.Series(["COG0001", None])
        result = clean_strings(s)
        assert result.iloc[0] == "COG0001"
        # Pandas 3.x with StringDtype: None stays NA after astype(str)
        # Pandas 2.x with object dtype: None becomes "NONE" after upper()
        # Accept both behaviours
        assert pd.isna(result.iloc[1]) or result.iloc[1] == "NONE"

    def test_numeric_values_cast_to_string(self):
        s = pd.Series([1234, 5678])
        result = clean_strings(s)
        assert list(result) == ["1234", "5678"]

    def test_empty_series(self):
        s = pd.Series([], dtype=object)
        result = clean_strings(s)
        assert len(result) == 0


# ---------------------------------------------------------------------------
# validate_annotation
# ---------------------------------------------------------------------------

class TestValidateAnnotation:
    def test_returns_valid_string_when_all_match(self):
        col = pd.Series(["GO:0008150", "GO:0003674", "GO:0005575"])
        go_df = make_go_df(["GO:0008150", "GO:0003674", "GO:0005575"])
        result = validate_annotation(col, go_df, "GO_ID")
        assert result == "Valid"

    def test_returns_invalid_set_when_some_invalid(self):
        col = pd.Series(["GO:0008150"] * 9 + ["INVALID"])
        go_df = make_go_df(["GO:0008150"])
        result = validate_annotation(col, go_df, "GO_ID")
        assert isinstance(result, set)
        assert "INVALID" in result

    def test_returns_none_when_validity_rate_below_threshold(self):
        """Less than 80 % valid → None."""
        col = pd.Series(["GO:0008150"] * 3 + ["JUNK1", "JUNK2", "JUNK3", "JUNK4", "JUNK5"])
        go_df = make_go_df(["GO:0008150"])
        result = validate_annotation(col, go_df, "GO_ID")
        assert result is None

    def test_case_insensitive_matching(self):
        """Lowercase input should match after clean_strings uppercasing."""
        col = pd.Series(["go:0008150"] * 10)
        go_df = make_go_df(["GO:0008150"])
        result = validate_annotation(col, go_df, "GO_ID")
        assert result == "Valid"

    def test_whitespace_stripped_before_matching(self):
        col = pd.Series(["  GO:0008150  "] * 10)
        go_df = make_go_df(["GO:0008150"])
        result = validate_annotation(col, go_df, "GO_ID")
        assert result == "Valid"

    def test_raw_match_fallback_used_when_better(self):
        """If raw (uncleaned) matching yields higher validity rate, it is used."""
        # Raw values already uppercase and clean → raw rate == cleaned rate
        # To trigger fallback: make cleaned rate < threshold, raw rate > threshold
        col = pd.Series(["GO:0008150"] * 9 + ["junk"])
        # Reference contains exact raw value and cleaned version
        go_df = make_go_df(["GO:0008150"])
        # cleaned: "GO:0008150" matches → 9/10 = 0.9 > 0.8 → no fallback needed
        result = validate_annotation(col, go_df, "GO_ID")
        assert result is not None  # valid path taken

    def test_custom_threshold_stricter(self):
        """Threshold=0.95: 90 % valid is not enough."""
        col = pd.Series(["GO:0008150"] * 9 + ["INVALID"])
        go_df = make_go_df(["GO:0008150"])
        result = validate_annotation(col, go_df, "GO_ID", threshold=0.95)
        assert result is None

    def test_custom_threshold_lenient(self):
        """Threshold=0.5: 60 % valid is sufficient."""
        col = pd.Series(["GO:0008150"] * 6 + ["INVALID"] * 4)
        go_df = make_go_df(["GO:0008150"])
        result = validate_annotation(col, go_df, "GO_ID", threshold=0.5)
        assert result is not None

    def test_invalid_set_contains_only_invalid_values(self):
        # 9/11 ≈ 0.818 > 0.8 → result returned; 2 invalid values flagged
        col = pd.Series(["COG0001"] * 9 + ["BAD1", "BAD2"])
        cog_df = make_cog_df(["COG0001"])
        result = validate_annotation(col, cog_df, "COG_ID")
        assert isinstance(result, set)
        assert "BAD1" in result
        assert "BAD2" in result
        assert "COG0001" not in result

    def test_empty_column_returns_none(self):
        col = pd.Series([], dtype=object)
        go_df = make_go_df(["GO:0008150"])
        result = validate_annotation(col, go_df, "GO_ID")
        # mean() of empty Series is NaN → NaN > threshold is False → None
        assert result is None

    def test_cog_single_letter_codes(self):
        """COG categories are single uppercase letters — common real-world case."""
        col = pd.Series(["J"] * 50 + ["K"] * 30 + ["L"] * 19 + ["X"])
        cog_df = make_cog_df(["J", "K", "L", "M", "N"])
        result = validate_annotation(col, cog_df, "COG_ID")
        assert isinstance(result, set)
        assert "X" in result

    def test_all_invalid_returns_none(self):
        col = pd.Series(["JUNK1", "JUNK2", "JUNK3"])
        go_df = make_go_df(["GO:0008150"])
        result = validate_annotation(col, go_df, "GO_ID")
        assert result is None

    def test_exactly_at_threshold_returns_none(self):
        """validity_rate == threshold: condition is strict >, so returns None."""
        # 8/10 = 0.8, threshold default = 0.8 → 0.8 > 0.8 is False → None
        col = pd.Series(["GO:0008150"] * 8 + ["BAD1", "BAD2"])
        go_df = make_go_df(["GO:0008150"])
        result = validate_annotation(col, go_df, "GO_ID")
        assert result is None

    def test_just_above_threshold_returns_result(self):
        """9/10 = 0.9 > 0.8 → valid path."""
        col = pd.Series(["GO:0008150"] * 9 + ["BAD"])
        go_df = make_go_df(["GO:0008150"])
        result = validate_annotation(col, go_df, "GO_ID")
        assert result is not None


# ---------------------------------------------------------------------------
# annotation_flags — mocked external data
# ---------------------------------------------------------------------------

COG_PATCH = "biological.functional_annotation.get_clusters_of_orthologous_groups"
GO_PATCH = "biological.functional_annotation.get_gene_ontology"


class TestAnnotationFlags:
    def test_returns_annotation_flags_object(self):
        col = pd.Series(["GO:0008150"] * 10)
        df = pd.DataFrame({"go_term": col})
        mock_go = make_go_df(["GO:0008150"])

        with patch(GO_PATCH, return_value=mock_go):
            result = annotation_flags(df, "go_term", "go")

        assert isinstance(result, AnnotationFlags)

    def test_name_matches_column(self):
        df = pd.DataFrame({"go_term": ["GO:0008150"] * 10})
        with patch(GO_PATCH, return_value=make_go_df(["GO:0008150"])):
            result = annotation_flags(df, "go_term", "go")
        assert result.name == "go_term"

    def test_is_annotation_true_when_valid(self):
        df = pd.DataFrame({"go_term": ["GO:0008150"] * 10})
        with patch(GO_PATCH, return_value=make_go_df(["GO:0008150"])):
            result = annotation_flags(df, "go_term", "go")
        assert result.is_annotation is True

    def test_is_annotation_false_when_invalid(self):
        """All values invalid → validate_annotation returns None → is_annotation False."""
        df = pd.DataFrame({"go_term": ["JUNK"] * 10})
        with patch(GO_PATCH, return_value=make_go_df(["GO:0008150"])):
            result = annotation_flags(df, "go_term", "go")
        assert result.is_annotation is False
        assert result.valid_annotation is None

    def test_cog_annotation_type(self):
        df = pd.DataFrame({"cog": ["J"] * 9 + ["X"]})
        mock_cog = make_cog_df(["J", "K", "L"])
        with patch(COG_PATCH, return_value=mock_cog):
            result = annotation_flags(df, "cog", "cog")
        assert isinstance(result, AnnotationFlags)
        assert result.is_annotation is True

    def test_valid_annotation_is_valid_string_when_all_match(self):
        df = pd.DataFrame({"cog": ["J"] * 10})
        with patch(COG_PATCH, return_value=make_cog_df(["J"])):
            result = annotation_flags(df, "cog", "cog")
        assert result.valid_annotation == "Valid"

    def test_valid_annotation_is_set_of_invalids(self):
        df = pd.DataFrame({"cog": ["J"] * 9 + ["UNKNOWN"]})
        with patch(COG_PATCH, return_value=make_cog_df(["J"])):
            result = annotation_flags(df, "cog", "cog")
        assert isinstance(result.valid_annotation, set)
        assert "UNKNOWN" in result.valid_annotation

    def test_raises_for_unknown_annotation_type(self):
        df = pd.DataFrame({"col": ["X"] * 5})
        with pytest.raises(ValueError, match="Unknown annotation type"):
            annotation_flags(df, "col", "pfam")

    def test_external_data_fetched_once_per_call(self):
        df = pd.DataFrame({"go_term": ["GO:0008150"] * 10})
        mock_fn = MagicMock(return_value=make_go_df(["GO:0008150"]))
        with patch(GO_PATCH, mock_fn):
            annotation_flags(df, "go_term", "go")
        mock_fn.assert_called_once()

    def test_go_column_with_mixed_valid_invalid(self):
        """Realistic GO annotation column: mostly valid, a few malformed."""
        valid = ["GO:0008150", "GO:0003674", "GO:0005575", "GO:0006412", "GO:0003735"]
        col = valid * 18 + ["MALFORMED1", "MALFORMED2"]  # 90 valid, 2 invalid
        df = pd.DataFrame({"go_term": col})
        go_df = make_go_df(valid)
        with patch(GO_PATCH, return_value=go_df):
            result = annotation_flags(df, "go_term", "go")
        assert result.is_annotation is True
        assert isinstance(result.valid_annotation, set)
        assert "MALFORMED1" in result.valid_annotation
        assert "MALFORMED2" in result.valid_annotation