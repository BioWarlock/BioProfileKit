"""Tests for cli.update_remote_data's `update` click command.

All four fetcher functions imported into this module (`get_tax_ids`,
`get_gene_ontology`, `get_clusters_of_orthologous_groups`,
`get_uniprot_swissprot_metadata`) are patched at their import site inside
`cli.update_remote_data`, since they themselves are network calls covered
(as far as is sensible) separately in test_remote_data.py. `print_step` and
`_fmt_duration` are likewise patched so these tests don't depend on the
real `cli.utils` implementation - only on this command's own control flow:
which flags trigger which fetcher, the `--all` shortcut, and the
"nothing selected" early return.
"""
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest
from click.testing import CliRunner

from cli.update_remote_data import update


@pytest.fixture
def runner():
    return CliRunner()


def _patched(**overrides):
    """Context-manager stack patching all four fetchers + print_step/_fmt_duration."""
    defaults = dict(
        get_tax_ids=MagicMock(return_value=pd.DataFrame({"a": range(3)})),
        get_gene_ontology=MagicMock(return_value=pd.DataFrame({"a": range(5)})),
        get_clusters_of_orthologous_groups=MagicMock(return_value=pd.DataFrame({"a": range(7)})),
        get_uniprot_swissprot_metadata=MagicMock(return_value=pd.DataFrame({"a": range(11)})),
    )
    defaults.update(overrides)
    return defaults


class TestFlagDispatch:
    def test_no_flags_prints_nothing_selected_and_returns(self, runner):
        mocks = _patched()
        with patch.multiple("cli.update_remote_data", **mocks), patch(
            "cli.update_remote_data.print_step"
        ) as mock_print_step:
            result = runner.invoke(update, [])

        assert result.exit_code == 0
        assert "Nothing selected" in result.output
        mocks["get_tax_ids"].assert_not_called()
        mocks["get_gene_ontology"].assert_not_called()
        mocks["get_clusters_of_orthologous_groups"].assert_not_called()
        mock_print_step.assert_not_called()

    def test_tax_flag_calls_only_get_tax_ids(self, runner):
        mocks = _patched()
        with patch.multiple("cli.update_remote_data", **mocks), patch(
            "cli.update_remote_data.print_step", return_value=MagicMock()
        ):
            result = runner.invoke(update, ["--tax"])

        assert result.exit_code == 0
        mocks["get_tax_ids"].assert_called_once()
        mocks["get_gene_ontology"].assert_not_called()
        mocks["get_clusters_of_orthologous_groups"].assert_not_called()
        mocks["get_uniprot_swissprot_metadata"].assert_not_called()

    def test_go_flag_calls_only_get_gene_ontology(self, runner):
        mocks = _patched()
        with patch.multiple("cli.update_remote_data", **mocks), patch(
            "cli.update_remote_data.print_step", return_value=MagicMock()
        ):
            result = runner.invoke(update, ["--go"])

        assert result.exit_code == 0
        mocks["get_gene_ontology"].assert_called_once()
        mocks["get_tax_ids"].assert_not_called()

    def test_cog_flag_calls_only_get_cog(self, runner):
        mocks = _patched()
        with patch.multiple("cli.update_remote_data", **mocks), patch(
            "cli.update_remote_data.print_step", return_value=MagicMock()
        ):
            result = runner.invoke(update, ["--cog"])

        assert result.exit_code == 0
        mocks["get_clusters_of_orthologous_groups"].assert_called_once()
        mocks["get_tax_ids"].assert_not_called()

    def test_uniprot_flag_alone_calls_only_uniprot(self, runner):
        mocks = _patched()
        with patch.multiple("cli.update_remote_data", **mocks), patch(
            "cli.update_remote_data.print_step", return_value=MagicMock()
        ):
            result = runner.invoke(update, ["--uniprot"])

        assert result.exit_code == 0
        assert "Nothing selected" not in result.output
        mocks["get_uniprot_swissprot_metadata"].assert_called_once()
        mocks["get_tax_ids"].assert_not_called()
        mocks["get_gene_ontology"].assert_not_called()
        mocks["get_clusters_of_orthologous_groups"].assert_not_called()

    def test_uniprot_flag_combined_with_another_flag_does_work(self, runner):
        mocks = _patched()
        with patch.multiple("cli.update_remote_data", **mocks), patch(
            "cli.update_remote_data.print_step", return_value=MagicMock()
        ):
            result = runner.invoke(update, ["--uniprot", "--tax"])

        assert result.exit_code == 0
        assert "Nothing selected" not in result.output
        mocks["get_uniprot_swissprot_metadata"].assert_called_once()
        mocks["get_tax_ids"].assert_called_once()

    def test_multiple_flags_call_multiple_fetchers(self, runner):
        mocks = _patched()
        with patch.multiple("cli.update_remote_data", **mocks), patch(
            "cli.update_remote_data.print_step", return_value=MagicMock()
        ):
            result = runner.invoke(update, ["--tax", "--go"])

        assert result.exit_code == 0
        mocks["get_tax_ids"].assert_called_once()
        mocks["get_gene_ontology"].assert_called_once()
        mocks["get_clusters_of_orthologous_groups"].assert_not_called()
        mocks["get_uniprot_swissprot_metadata"].assert_not_called()

    def test_all_flag_calls_all_four_fetchers(self, runner):
        mocks = _patched()
        with patch.multiple("cli.update_remote_data", **mocks), patch(
            "cli.update_remote_data.print_step", return_value=MagicMock()
        ):
            result = runner.invoke(update, ["--all"])

        assert result.exit_code == 0
        mocks["get_tax_ids"].assert_called_once()
        mocks["get_gene_ontology"].assert_called_once()
        mocks["get_clusters_of_orthologous_groups"].assert_called_once()
        mocks["get_uniprot_swissprot_metadata"].assert_called_once()


class TestForceRefresh:
    def test_force_refresh_propagated_to_all_active_fetchers(self, runner):
        mocks = _patched()
        with patch.multiple("cli.update_remote_data", **mocks), patch(
            "cli.update_remote_data.print_step", return_value=MagicMock()
        ):
            result = runner.invoke(update, ["--all", "--force-refresh"])

        assert result.exit_code == 0
        mocks["get_tax_ids"].assert_called_once_with(force_refresh=True)
        mocks["get_gene_ontology"].assert_called_once_with(force_refresh=True)
        mocks["get_clusters_of_orthologous_groups"].assert_called_once_with(force_refresh=True)
        mocks["get_uniprot_swissprot_metadata"].assert_called_once_with(force_refresh=True)

    def test_force_refresh_defaults_to_false(self, runner):
        mocks = _patched()
        with patch.multiple("cli.update_remote_data", **mocks), patch(
            "cli.update_remote_data.print_step", return_value=MagicMock()
        ):
            result = runner.invoke(update, ["--tax"])

        mocks["get_tax_ids"].assert_called_once_with(force_refresh=False)


class TestOutput:
    def test_step_labels_and_counts_reported(self, runner):
        mocks = _patched(
            get_tax_ids=MagicMock(return_value=pd.DataFrame({"a": range(3)})),
        )
        done_mock = MagicMock()
        with patch.multiple("cli.update_remote_data", **mocks), patch(
            "cli.update_remote_data.print_step", return_value=done_mock
        ) as mock_print_step:
            result = runner.invoke(update, ["--tax"])

        assert result.exit_code == 0
        mock_print_step.assert_called_once_with("NCBI Taxonomy")
        done_mock.assert_called_once_with("3 entries")

    def test_completion_message_includes_output_dir(self, runner):
        mocks = _patched()
        with patch.multiple("cli.update_remote_data", **mocks), patch(
            "cli.update_remote_data.print_step", return_value=MagicMock()
        ), patch("cli.update_remote_data._fmt_duration", return_value="0.1s"):
            result = runner.invoke(update, ["--tax"])

        assert result.exit_code == 0
        assert "Download completed in 0.1s" in result.output

    def test_command_does_not_raise_on_success(self, runner):
        mocks = _patched()
        with patch.multiple("cli.update_remote_data", **mocks), patch(
            "cli.update_remote_data.print_step", return_value=MagicMock()
        ):
            result = runner.invoke(update, ["--all"])

        assert result.exception is None

    def test_help_option_lists_all_flags(self, runner):
        result = runner.invoke(update, ["--help"])

        assert result.exit_code == 0
        for flag in ["--tax", "--go", "--cog", "--uniprot", "--all", "--force-refresh"]:
            assert flag in result.output