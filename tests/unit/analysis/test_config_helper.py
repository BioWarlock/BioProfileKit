"""Tests for analysis.plot_utils._config, the per-plot Plotly config factory."""

from analysis.plot_utils import _config


def test_sets_export_filename():
    assert _config("gc_distribution")["toImageButtonOptions"]["filename"] == "gc_distribution"


def test_returns_independent_dicts():
    # Each plot must get its own config; mutating one must not leak into another
    # (the old shared module-level dict had exactly that problem).
    first = _config("length_distribution")
    second = _config("at_gc_skewness")
    first["toImageButtonOptions"]["filename"] = "changed"
    assert second["toImageButtonOptions"]["filename"] == "at_gc_skewness"
    assert _config("length_distribution")["toImageButtonOptions"]["filename"] == "length_distribution"
