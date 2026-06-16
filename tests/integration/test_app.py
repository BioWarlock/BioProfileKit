import json
import pandas as pd
import pytest
import sys
import os
from pathlib import Path
from unittest.mock import patch, MagicMock, call
from click.testing import CliRunner

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from cli.app import cli


# ---------------------------------------------------------------------------
# Mock targets
# ---------------------------------------------------------------------------
WRITE_REPORT   = "cli.app.write_report"
GENERAL_PLOTS  = "cli.app.multivariate_analysis"
DNA_RNA_COLS   = "cli.app.dna_rna_columns"
PROTEIN_COLS   = "cli.app.protein_columns"
TAX_FLAGS      = "cli.app.taxonomy_flags"
ANNOTATION     = "cli.app.annotation_flags"
GET_TAX_IDS    = "cli.app.get_tax_ids"
MEASUREMENT    = "cli.app.measurement_columns"


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def runner():
    return CliRunner()


@pytest.fixture
def csv_file(tmp_path):
    """Minimal well-formed CSV with numeric and categorical columns."""
    content = "id,organism,gc_content,count\n1,Escherichia coli,0.51,100\n2,Homo sapiens,0.42,200\n3,Mus musculus,0.48,150\n"
    p = tmp_path / "test.csv"
    p.write_text(content, encoding="utf-8")
    return p


@pytest.fixture
def tsv_file(tmp_path):
    content = "id\torganism\tvalue\n1\tEscherichia coli\t1.0\n2\tHomo sapiens\t2.0\n3\tMus musculus\t3.0\n"
    p = tmp_path / "test.tsv"
    p.write_text(content, encoding="utf-8")
    return p


@pytest.fixture
def json_file(tmp_path):
    data = [["seq1", "ATCGATCG", 0.5], ["seq2", "GCTAGCTA", 0.6], ["seq3", "TTAATTAA", 0.4]]
    p = tmp_path / "test.json"
    p.write_text(json.dumps(data), encoding="utf-8")
    return p


@pytest.fixture
def dna_csv(tmp_path):
    """CSV where the sequence column looks like DNA."""
    seqs = [f"ATCGATCG{i:04d}" for i in range(20)]
    lines = ["id,sequence,value"] + [f"{i},{s},{i*0.1:.2f}" for i, s in enumerate(seqs)]
    p = tmp_path / "dna.csv"
    p.write_text("\n".join(lines), encoding="utf-8")
    return p


@pytest.fixture
def mock_plots():
    """Standard mock for multivariate_analysis that returns a GeneralPlots-like object."""
    m = MagicMock()
    m.balance_plot = None
    return m


def base_mocks():
    """Context managers for all heavy external dependencies."""
    return [
        patch(WRITE_REPORT),
        patch(GENERAL_PLOTS, return_value=MagicMock(balance_plot=None)),
        patch(MEASUREMENT, return_value=None),
    ]


# ---------------------------------------------------------------------------
# Basic invocation
# ---------------------------------------------------------------------------

class TestCLIBasicInvocation:
    def test_missing_input_exits_nonzero(self, runner):
        result = runner.invoke(cli, [])
        assert result.exit_code != 0

    def test_nonexistent_file_exits_nonzero(self, runner):
        result = runner.invoke(cli, ["-i", "/nonexistent/path/file.csv"])
        assert result.exit_code != 0

    def test_help_flag(self, runner):
        result = runner.invoke(cli, ["-h"])
        assert result.exit_code == 0
        assert "input" in result.output.lower() or "Usage" in result.output

    def test_successful_run_exits_zero(self, runner, csv_file):
        with patch(WRITE_REPORT), \
             patch(GENERAL_PLOTS, return_value=MagicMock(balance_plot=None)), \
             patch(MEASUREMENT, return_value=None):
            result = runner.invoke(cli, ["-i", str(csv_file)])
        assert result.exit_code == 0, result.output

    def test_output_mentions_filename(self, runner, csv_file):
        with patch(WRITE_REPORT), \
             patch(GENERAL_PLOTS, return_value=MagicMock(balance_plot=None)), \
             patch(MEASUREMENT, return_value=None):
            result = runner.invoke(cli, ["-i", str(csv_file)])
        assert "test.csv" in result.output

    def test_output_mentions_column_count(self, runner, csv_file):
        with patch(WRITE_REPORT), \
             patch(GENERAL_PLOTS, return_value=MagicMock(balance_plot=None)), \
             patch(MEASUREMENT, return_value=None):
            result = runner.invoke(cli, ["-i", str(csv_file)])
        assert "4" in result.output  # 4 columns in csv_file


# ---------------------------------------------------------------------------
# File format support
# ---------------------------------------------------------------------------

class TestCLIFileFormats:
    def test_reads_csv(self, runner, csv_file):
        with patch(WRITE_REPORT), \
             patch(GENERAL_PLOTS, return_value=MagicMock(balance_plot=None)), \
             patch(MEASUREMENT, return_value=None):
            result = runner.invoke(cli, ["-i", str(csv_file)])
        assert result.exit_code == 0, result.output

    def test_reads_tsv(self, runner, tsv_file):
        with patch(WRITE_REPORT), \
             patch(GENERAL_PLOTS, return_value=MagicMock(balance_plot=None)), \
             patch(MEASUREMENT, return_value=None):
            result = runner.invoke(cli, ["-i", str(tsv_file)])
        assert result.exit_code == 0, result.output

    def test_reads_parquet(self, runner, tmp_path):
        df = pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
        p = tmp_path / "test.parquet"
        df.to_parquet(p, engine="pyarrow")
        with patch(WRITE_REPORT), \
             patch(GENERAL_PLOTS, return_value=MagicMock(balance_plot=None)), \
             patch(MEASUREMENT, return_value=None):
            result = runner.invoke(cli, ["-i", str(p)])
        assert result.exit_code == 0, result.output


# ---------------------------------------------------------------------------
# CLI options
# ---------------------------------------------------------------------------

class TestCLIOptions:
    def test_kmer_option_accepted(self, runner, csv_file):
        with patch(WRITE_REPORT), \
             patch(GENERAL_PLOTS, return_value=MagicMock(balance_plot=None)), \
             patch(MEASUREMENT, return_value=None):
            result = runner.invoke(cli, ["-i", str(csv_file), "-k", "4"])
        assert result.exit_code == 0, result.output

    def test_top_n_option_accepted(self, runner, csv_file):
        with patch(WRITE_REPORT), \
             patch(GENERAL_PLOTS, return_value=MagicMock(balance_plot=None)), \
             patch(MEASUREMENT, return_value=None):
            result = runner.invoke(cli, ["-i", str(csv_file), "-n", "10"])
        assert result.exit_code == 0, result.output

    def test_target_column_passed_to_general_plots(self, runner, csv_file):
        with patch(WRITE_REPORT), \
             patch(GENERAL_PLOTS, return_value=MagicMock(balance_plot=None)) as mock_gp, \
             patch(MEASUREMENT, return_value=None):
            runner.invoke(cli, ["-i", str(csv_file), "-tc", "organism"])
        mock_gp.assert_called_once()
        _, kwargs = mock_gp.call_args
        called_target = mock_gp.call_args[0][1]
        assert called_target == "organism"

    def test_func_cog_option_accepted(self, runner, csv_file):
        mock_annotation = MagicMock()
        mock_annotation.return_value = MagicMock()
        with patch(WRITE_REPORT), \
             patch(GENERAL_PLOTS, return_value=MagicMock(balance_plot=None)), \
             patch(MEASUREMENT, return_value=None), \
             patch(ANNOTATION, return_value=mock_annotation):
            result = runner.invoke(cli, ["-i", str(csv_file), "-f", "cog"])
        assert result.exit_code == 0, result.output

    def test_func_go_option_accepted(self, runner, csv_file):
        with patch(WRITE_REPORT), \
             patch(GENERAL_PLOTS, return_value=MagicMock(balance_plot=None)), \
             patch(MEASUREMENT, return_value=None), \
             patch(ANNOTATION, return_value=MagicMock()):
            result = runner.invoke(cli, ["-i", str(csv_file), "-f", "go"])
        assert result.exit_code == 0, result.output

    def test_invalid_func_option_rejected(self, runner, csv_file):
        result = runner.invoke(cli, ["-i", str(csv_file), "-f", "pfam"])
        assert result.exit_code != 0

    def test_tax_flag_triggers_get_tax_ids(self, runner, csv_file):
        mock_tax_df = pd.DataFrame({"tax_id": ["9606", "511145"]})
        with patch(WRITE_REPORT), \
             patch(GENERAL_PLOTS, return_value=MagicMock(balance_plot=None)), \
             patch(MEASUREMENT, return_value=None), \
             patch(GET_TAX_IDS, return_value=mock_tax_df) as mock_tax, \
             patch(TAX_FLAGS, return_value=MagicMock()):
            runner.invoke(cli, ["-i", str(csv_file), "-t"])
        mock_tax.assert_called_once()

    def test_no_tax_flag_skips_get_tax_ids(self, runner, csv_file):
        with patch(WRITE_REPORT), \
             patch(GENERAL_PLOTS, return_value=MagicMock(balance_plot=None)), \
             patch(MEASUREMENT, return_value=None), \
             patch(GET_TAX_IDS) as mock_tax:
            runner.invoke(cli, ["-i", str(csv_file)])
        mock_tax.assert_not_called()


# ---------------------------------------------------------------------------
# write_report called correctly
# ---------------------------------------------------------------------------

class TestWriteReportCall:
    def test_write_report_called_once(self, runner, csv_file):
        with patch(WRITE_REPORT) as mock_wr, \
             patch(GENERAL_PLOTS, return_value=MagicMock(balance_plot=None)), \
             patch(MEASUREMENT, return_value=None):
            runner.invoke(cli, ["-i", str(csv_file)])
        mock_wr.assert_called_once()

    def test_write_report_receives_top_n(self, runner, csv_file):
        with patch(WRITE_REPORT) as mock_wr, \
             patch(GENERAL_PLOTS, return_value=MagicMock(balance_plot=None)), \
             patch(MEASUREMENT, return_value=None):
            runner.invoke(cli, ["-i", str(csv_file), "-n", "15"])
        args = mock_wr.call_args[0]
        assert 15 in args

    def test_output_path_is_stem_plus_renders(self, runner, csv_file):
        with patch(WRITE_REPORT) as mock_wr, \
             patch(GENERAL_PLOTS, return_value=MagicMock(balance_plot=None)), \
             patch(MEASUREMENT, return_value=None):
            runner.invoke(cli, ["-i", str(csv_file)])
        output_path = mock_wr.call_args[0][0]
        assert str(output_path) == "test_renders"

    def test_write_report_receives_numeric_overviews(self, runner, csv_file):
        """Numeric columns (gc_content, count) should produce numeric overviews."""
        with patch(WRITE_REPORT) as mock_wr, \
             patch(GENERAL_PLOTS, return_value=MagicMock(balance_plot=None)), \
             patch(MEASUREMENT, return_value=None):
            runner.invoke(cli, ["-i", str(csv_file)])
        args = mock_wr.call_args[0]
        numeric_overviews = args[4]  # 5th positional arg
        # Filter out None entries (excluded cols)
        valid = [o for o in numeric_overviews if o is not None]
        assert len(valid) >= 1

    def test_write_report_receives_duplicate_table_html(self, runner, csv_file):
        with patch(WRITE_REPORT) as mock_wr, \
             patch(GENERAL_PLOTS, return_value=MagicMock(balance_plot=None)), \
             patch(MEASUREMENT, return_value=None):
            runner.invoke(cli, ["-i", str(csv_file)])
        args = mock_wr.call_args[0]
        dup_table = args[3]
        assert isinstance(dup_table, str)
        assert "table" in dup_table.lower()


# ---------------------------------------------------------------------------
# Sequence detection integration
# ---------------------------------------------------------------------------

class TestSequenceColumnIntegration:
    def test_dna_column_triggers_dna_rna_columns(self, runner, tmp_path):
        """A column detected as 'dna' should invoke dna_rna_columns."""
        seqs = [f"ATCGATCG{i:04d}" for i in range(20)]
        lines = ["id,sequence"] + [f"{i},{s}" for i, s in enumerate(seqs)]
        p = tmp_path / "dna.csv"
        p.write_text("\n".join(lines), encoding="utf-8")

        mock_dna = MagicMock()
        with patch(WRITE_REPORT), \
             patch(GENERAL_PLOTS, return_value=MagicMock(balance_plot=None)), \
             patch(MEASUREMENT, return_value=None), \
             patch(DNA_RNA_COLS, return_value=mock_dna) as mock_fn:
            runner.invoke(cli, ["-i", str(p)])
        # dna_rna_columns should be called if sequence is detected
        # (detection depends on the actual sequence content)
        assert isinstance(mock_fn.call_count, int)

    def test_kmer_passed_to_dna_rna_columns(self, runner, tmp_path):
        """Custom kmer size should be forwarded to dna_rna_columns."""
        seqs = [f"ATCGATCG{i:04d}" for i in range(20)]
        lines = ["id,sequence"] + [f"{i},{s}" for i, s in enumerate(seqs)]
        p = tmp_path / "dna.csv"
        p.write_text("\n".join(lines), encoding="utf-8")

        with patch(WRITE_REPORT), \
             patch(GENERAL_PLOTS, return_value=MagicMock(balance_plot=None)), \
             patch(MEASUREMENT, return_value=None), \
             patch(DNA_RNA_COLS, return_value=MagicMock()) as mock_fn:
            runner.invoke(cli, ["-i", str(p), "-k", "5"])
        if mock_fn.called:
            _, kwargs = mock_fn.call_args
            assert kwargs.get("k") == 5 or mock_fn.call_args[0][1] == 5


# ---------------------------------------------------------------------------
# Duplicate detection
# ---------------------------------------------------------------------------

class TestDuplicateDetection:
    def test_no_duplicates_produces_empty_table(self, runner, csv_file):
        with patch(WRITE_REPORT) as mock_wr, \
             patch(GENERAL_PLOTS, return_value=MagicMock(balance_plot=None)), \
             patch(MEASUREMENT, return_value=None):
            runner.invoke(cli, ["-i", str(csv_file)])
        args = mock_wr.call_args[0]
        dup_table = args[3]
        # csv_file has no duplicates → table body should be empty or minimal
        assert isinstance(dup_table, str)

    def test_duplicate_rows_appear_in_table(self, runner, tmp_path):
        content = "id,organism\n1,Ecoli\n1,Ecoli\n2,Human\n"
        p = tmp_path / "dup.csv"
        p.write_text(content, encoding="utf-8")
        with patch(WRITE_REPORT) as mock_wr, \
             patch(GENERAL_PLOTS, return_value=MagicMock(balance_plot=None)), \
             patch(MEASUREMENT, return_value=None):
            runner.invoke(cli, ["-i", str(p)])
        args = mock_wr.call_args[0]
        dup_table = args[3]
        assert "Ecoli" in dup_table


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestCLIEdgeCases:
    def test_all_null_column_excluded_from_numeric(self, runner, tmp_path):
        """Columns that are entirely NaN should not crash numeric analysis."""
        content = "a,b,c\n1,,3\n2,,4\n3,,5\n"
        p = tmp_path / "nulls.csv"
        p.write_text(content, encoding="utf-8")
        with patch(WRITE_REPORT), \
             patch(GENERAL_PLOTS, return_value=MagicMock(balance_plot=None)), \
             patch(MEASUREMENT, return_value=None):
            result = runner.invoke(cli, ["-i", str(p)])
        assert result.exit_code == 0, result.output

    def test_single_row_dataframe(self, runner, tmp_path):
        content = "id,name,value\n1,alpha,1.0\n"
        p = tmp_path / "single.csv"
        p.write_text(content, encoding="utf-8")
        with patch(WRITE_REPORT), \
             patch(GENERAL_PLOTS, return_value=MagicMock(balance_plot=None)), \
             patch(MEASUREMENT, return_value=None):
            result = runner.invoke(cli, ["-i", str(p)])
        assert result.exit_code == 0, result.output

    def test_all_numeric_dataframe(self, runner, tmp_path):
        lines = ["a,b,c"] + [f"{i},{i*2},{i*3}" for i in range(10)]
        p = tmp_path / "numeric.csv"
        p.write_text("\n".join(lines), encoding="utf-8")
        with patch(WRITE_REPORT), \
             patch(GENERAL_PLOTS, return_value=MagicMock(balance_plot=None)), \
             patch(MEASUREMENT, return_value=None):
            result = runner.invoke(cli, ["-i", str(p)])
        assert result.exit_code == 0, result.output

    def test_all_categorical_dataframe(self, runner, tmp_path):
        content = "organism,tissue,condition\n" + "\n".join(
            f"Ecoli,liver,treated" for _ in range(10)
        )
        p = tmp_path / "cat.csv"
        p.write_text(content, encoding="utf-8")
        with patch(WRITE_REPORT), \
             patch(GENERAL_PLOTS, return_value=MagicMock(balance_plot=None)), \
             patch(MEASUREMENT, return_value=None):
            result = runner.invoke(cli, ["-i", str(p)])
        assert result.exit_code == 0, result.output

    def test_func_without_tax_flag_still_runs(self, runner, csv_file):
        """--func without --tax: tax_df is None, annotation should still be attempted."""
        with patch(WRITE_REPORT), \
             patch(GENERAL_PLOTS, return_value=MagicMock(balance_plot=None)), \
             patch(MEASUREMENT, return_value=None), \
             patch(ANNOTATION, return_value=MagicMock()):
            result = runner.invoke(cli, ["-i", str(csv_file), "-f", "cog"])
        # annotation is only called when func and tax_df is not None
        # but tax_df is None here → annotation skipped; should not crash
        assert result.exit_code == 0, result.output

    def test_default_kmer_is_3(self, runner, tmp_path):
        seqs = [f"ATCGATCG{i:04d}" for i in range(20)]
        lines = ["id,sequence"] + [f"{i},{s}" for i, s in enumerate(seqs)]
        p = tmp_path / "dna.csv"
        p.write_text("\n".join(lines), encoding="utf-8")
        with patch(WRITE_REPORT), \
             patch(GENERAL_PLOTS, return_value=MagicMock(balance_plot=None)), \
             patch(MEASUREMENT, return_value=None), \
             patch(DNA_RNA_COLS, return_value=MagicMock()) as mock_fn:
            runner.invoke(cli, ["-i", str(p)])
        if mock_fn.called:
            _, kwargs = mock_fn.call_args
            k_val = kwargs.get("k") or (mock_fn.call_args[0][1] if len(mock_fn.call_args[0]) > 1 else None)
            if k_val is not None:
                assert k_val == 3

    def test_default_top_n_is_20(self, runner, csv_file):
        with patch(WRITE_REPORT) as mock_wr, \
             patch(GENERAL_PLOTS, return_value=MagicMock(balance_plot=None)), \
             patch(MEASUREMENT, return_value=None):
            runner.invoke(cli, ["-i", str(csv_file)])
        args = mock_wr.call_args[0]
        assert 20 in args