import pytest
import sys
import os
from pathlib import Path
from unittest.mock import patch, MagicMock, mock_open
from jinja2 import Environment, DictLoader

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from cli.report_writer import write_report, _render_to_file


# ---------------------------------------------------------------------------
# Minimal Jinja2 environment with in-memory templates
# No filesystem or importlib_resources dependency needed
# ---------------------------------------------------------------------------

TEMPLATES = {
    "LandingPage.jinja": "<html><body>Landing</body></html>",
    "numeric_overview.jinja": "<html><body>{{ general }}{{ dup_groups }}</body></html>",
    "columns.jinja": "<html><body>{{ columns }}{{ overview }}{{ categorical }}{{ top_n }}</body></html>",
    "general_statistics.jinja": "<html><body>{{ plots|safe }}</body></html>",
}

TEST_ENV = Environment(loader=DictLoader(TEMPLATES), autoescape=True)

ENV_PATCH = "cli.report_writer.env"
STATIC_PATCH = "cli.report_writer.STATIC_DIR"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def invoke_write_report(tmp_path, general="general", plots="plots",
                        dup_groups="<table></table>", column_overviews=None,
                        numeric_overviews=None, categorical_overviews=None,
                        top_n=20, quality=None):
    """Call write_report with mocked env and static dir."""
    output = tmp_path / "report"
    with patch(ENV_PATCH, TEST_ENV), \
         patch(STATIC_PATCH, MagicMock(__str__=lambda s: str(tmp_path / "static_src"))):
        # Create a dummy static source so copytree doesn't fail
        (tmp_path / "static_src").mkdir(exist_ok=True)
        (tmp_path / "static_src" / "style.css").write_text("body{}")
        write_report(
            output,
            general=general,
            plots=plots,
            dup_groups=dup_groups,
            column_overviews=column_overviews or [],
            numeric_overviews=numeric_overviews or [],
            categorical_overviews=categorical_overviews or [],
            top_n=top_n,
            quality=quality,
        )
    return output


# ---------------------------------------------------------------------------
# Output directory creation
# ---------------------------------------------------------------------------

class TestOutputDirectory:
    def test_creates_output_directory(self, tmp_path):
        output = invoke_write_report(tmp_path)
        assert output.exists()
        assert output.is_dir()

    def test_creates_nested_output_directory(self, tmp_path):
        """parents=True: deeply nested path should be created."""
        nested = tmp_path / "a" / "b" / "c" / "report"
        with patch(ENV_PATCH, TEST_ENV), \
             patch(STATIC_PATCH, MagicMock(__str__=lambda s: str(tmp_path / "static_src"))):
            (tmp_path / "static_src").mkdir(exist_ok=True)
            write_report(nested, "g", MagicMock(), "<table/>", [], [], [], 20)
        assert nested.exists()

    def test_existing_directory_not_overwritten(self, tmp_path):
        """exist_ok=True: calling twice should not raise."""
        invoke_write_report(tmp_path)
        invoke_write_report(tmp_path)  # second call on same path

    def test_static_dir_copied(self, tmp_path):
        invoke_write_report(tmp_path)
        assert (tmp_path / "report" / "static").exists()
        assert (tmp_path / "report" / "static" / "style.css").exists()

    def test_static_dir_merges_with_existing(self, tmp_path):
        """dirs_exist_ok=True: static already present should not raise."""
        output = invoke_write_report(tmp_path)
        (output / "static").mkdir(exist_ok=True)
        invoke_write_report(tmp_path)  # should not raise


# ---------------------------------------------------------------------------
# Output files created
# ---------------------------------------------------------------------------

class TestOutputFilesCreated:
    def test_index_html_created(self, tmp_path):
        output = invoke_write_report(tmp_path)
        assert (output / "index.html").exists()

    def test_numeric_data_html_created(self, tmp_path):
        output = invoke_write_report(tmp_path)
        assert (output / "numeric_data.html").exists()

    def test_columns_html_created(self, tmp_path):
        output = invoke_write_report(tmp_path)
        assert (output / "columns.html").exists()

    def test_general_statistics_html_created(self, tmp_path):
        output = invoke_write_report(tmp_path)
        assert (output / "general_statistics.html").exists()

    def test_all_four_html_files_created(self, tmp_path):
        output = invoke_write_report(tmp_path)
        expected = {"index.html", "numeric_data.html", "columns.html", "general_statistics.html"}
        created = {f.name for f in output.glob("*.html")}
        assert expected.issubset(created)

    def test_files_are_utf8_encoded(self, tmp_path):
        output = invoke_write_report(tmp_path)
        for html in output.glob("*.html"):
            content = html.read_text(encoding="utf-8")
            assert isinstance(content, str)


# ---------------------------------------------------------------------------
# Template context passing
# ---------------------------------------------------------------------------

class TestTemplateContext:
    def test_general_passed_to_numeric_template(self, tmp_path):
        output = invoke_write_report(tmp_path, general="MY_GENERAL_MARKER")
        content = (output / "numeric_data.html").read_text()
        assert "MY_GENERAL_MARKER" in content

    def test_dup_groups_passed_to_numeric_template(self, tmp_path):
        output = invoke_write_report(tmp_path, dup_groups="<table>DUP_MARKER</table>")
        content = (output / "numeric_data.html").read_text()
        assert "DUP_MARKER" in content

    def test_plots_passed_to_general_statistics_template(self, tmp_path):
        output = invoke_write_report(tmp_path, plots="PLOTS_MARKER")
        content = (output / "general_statistics.html").read_text()
        assert "PLOTS_MARKER" in content

    def test_top_n_passed_to_columns_template(self, tmp_path):
        output = invoke_write_report(tmp_path, top_n=42)
        content = (output / "columns.html").read_text()
        assert "42" in content

    def test_landing_page_has_no_context(self, tmp_path):
        """LandingPage.jinja receives no context variables — should still render."""
        output = invoke_write_report(tmp_path)
        content = (output / "index.html").read_text()
        assert "Landing" in content

    def test_empty_column_overviews_renders_without_crash(self, tmp_path):
        output = invoke_write_report(tmp_path, column_overviews=[])
        assert (output / "columns.html").exists()

    def test_none_entries_in_numeric_overviews_passed_through(self, tmp_path):
        """numeric_overviews may contain None for excluded columns."""
        output = invoke_write_report(tmp_path, numeric_overviews=[None, MagicMock(), None])
        assert (output / "columns.html").exists()


# ---------------------------------------------------------------------------
# _render_to_file
# ---------------------------------------------------------------------------

class TestRenderToFile:
    def test_creates_file_at_given_path(self, tmp_path):
        filepath = tmp_path / "out.html"
        with patch(ENV_PATCH, TEST_ENV):
            _render_to_file(filepath, "LandingPage.jinja")
        assert filepath.exists()

    def test_file_content_matches_template(self, tmp_path):
        filepath = tmp_path / "out.html"
        with patch(ENV_PATCH, TEST_ENV):
            _render_to_file(filepath, "LandingPage.jinja")
        assert "Landing" in filepath.read_text(encoding="utf-8")

    def test_context_variables_rendered(self, tmp_path):
        filepath = tmp_path / "out.html"
        with patch(ENV_PATCH, TEST_ENV):
            _render_to_file(filepath, "numeric_overview.jinja",
                            general="TEST_GENERAL", dup_groups="TEST_DUPS")
        content = filepath.read_text(encoding="utf-8")
        assert "TEST_GENERAL" in content
        assert "TEST_DUPS" in content

    def test_overwrites_existing_file(self, tmp_path):
        filepath = tmp_path / "out.html"
        filepath.write_text("OLD CONTENT", encoding="utf-8")
        with patch(ENV_PATCH, TEST_ENV):
            _render_to_file(filepath, "LandingPage.jinja")
        assert "OLD CONTENT" not in filepath.read_text(encoding="utf-8")

    def test_file_written_as_utf8(self, tmp_path):
        filepath = tmp_path / "out.html"
        with patch(ENV_PATCH, TEST_ENV):
            _render_to_file(filepath, "LandingPage.jinja")
        # Should be readable as UTF-8 without errors
        content = filepath.read_text(encoding="utf-8")
        assert len(content) > 0

    def test_missing_template_raises(self, tmp_path):
        from jinja2 import TemplateNotFound
        filepath = tmp_path / "out.html"
        with patch(ENV_PATCH, TEST_ENV), \
             pytest.raises(TemplateNotFound):
            _render_to_file(filepath, "NonExistent.jinja")

    def test_parent_directory_must_exist(self, tmp_path):
        """_render_to_file does not create parent dirs — they must already exist."""
        filepath = tmp_path / "nonexistent_subdir" / "out.html"
        with patch(ENV_PATCH, TEST_ENV), \
             pytest.raises(FileNotFoundError):
            _render_to_file(filepath, "LandingPage.jinja")


# ---------------------------------------------------------------------------
# Autoescaping
# ---------------------------------------------------------------------------

class TestAutoescaping:
    def test_html_in_general_context_is_escaped(self, tmp_path):
        """autoescape=True: raw HTML in general (non-safe) variables is escaped."""
        output = invoke_write_report(
            tmp_path,
            general="<script>alert('xss')</script>",
        )
        content = (output / "numeric_data.html").read_text()
        assert "<script>" not in content
        assert "&lt;script&gt;" in content

    def test_duplicate_groups_content_is_escaped(self, tmp_path):
        """dup_groups no longer uses |safe (real template renders structured
        row values via plain {{ }} interpolation) — raw HTML should now be
        escaped, same as any other non-safe context variable."""
        output = invoke_write_report(
            tmp_path,
            dup_groups="<table><tr><td>data</td></tr></table>",
        )
        content = (output / "numeric_data.html").read_text()
        assert "<table>" not in content
        assert "&lt;table&gt;" in content

    def test_plots_rendered_unescaped(self, tmp_path):
        """plots uses |safe — Plotly HTML renders correctly."""
        output = invoke_write_report(
            tmp_path,
            plots="<div class='plotly-graph-div'>CHART</div>",
        )
        content = (output / "general_statistics.html").read_text()
        assert "<div" in content
        assert "CHART" in content
        assert "&lt;div" not in content