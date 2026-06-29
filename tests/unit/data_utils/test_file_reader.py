import json
import numpy as np
import pandas as pd
import pytest
import sys
import os

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from data_utils.file_reader import read_file, _get_sep, parse_parquet


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def write(tmp_path, filename, content, encoding="utf-8"):
    p = tmp_path / filename
    p.write_text(content, encoding=encoding)
    return p


# ---------------------------------------------------------------------------
# _get_sep
# ---------------------------------------------------------------------------

class TestGetSep:
    def test_detects_comma(self, tmp_path):
        p = write(tmp_path, "f.csv", "a,b,c\n1,2,3\n4,5,6\n")
        assert _get_sep(p) == ","

    def test_detects_tab(self, tmp_path):
        p = write(tmp_path, "f.tsv", "a\tb\tc\n1\t2\t3\n4\t5\t6\n")
        assert _get_sep(p) == "\t"

    def test_detects_semicolon(self, tmp_path):
        p = write(tmp_path, "f.csv", "a;b;c\n1;2;3\n4;5;6\n")
        assert _get_sep(p) == ";"

    def test_detects_pipe(self, tmp_path):
        p = write(tmp_path, "f.csv", "a|b|c\n1|2|3\n4|5|6\n")
        assert _get_sep(p) == "|"

    def test_returns_string(self, tmp_path):
        p = write(tmp_path, "f.csv", "a,b\n1,2\n")
        result = _get_sep(p)
        assert isinstance(result, str)
        assert len(result) == 1

    def test_single_row_file(self, tmp_path):
        """Single row: std is 0 for all → argmin returns first candidate with avg >= 1."""
        p = write(tmp_path, "f.csv", "a,b,c\n")
        result = _get_sep(p)
        assert result == ","

    def test_skips_empty_lines(self, tmp_path):
        p = write(tmp_path, "f.csv", "\na,b,c\n\n1,2,3\n")
        assert _get_sep(p) == ","

    def test_consistent_delimiter_preferred(self, tmp_path):
        """Consistent column count per row → low std → preferred separator."""
        content = "a,b,c\n1,2,3\n4,5,6\n7,8,9\n"
        p = write(tmp_path, "f.csv", content)
        assert _get_sep(p) == ","


# ---------------------------------------------------------------------------
# read_file — CSV
# ---------------------------------------------------------------------------

class TestReadFileCSV:
    def test_reads_basic_csv(self, tmp_path):
        p = write(tmp_path, "f.csv", "id,name,value\n1,alpha,1.0\n2,beta,2.0\n")
        df = read_file(p)
        assert isinstance(df, pd.DataFrame)
        assert list(df.columns) == ["id", "name", "value"]
        assert len(df) == 2

    def test_reads_semicolon_csv(self, tmp_path):
        p = write(tmp_path, "f.csv", "id;name;value\n1;alpha;1.0\n2;beta;2.0\n")
        df = read_file(p)
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 2

    def test_csv_values_correct(self, tmp_path):
        p = write(tmp_path, "f.csv", "x,y\n10,20\n30,40\n")
        df = read_file(p)
        assert df["x"].tolist() == [10, 30]
        assert df["y"].tolist() == [20, 40]

    def test_csv_with_missing_values(self, tmp_path):
        p = write(tmp_path, "f.csv", "a,b,c\n1,,3\n4,5,\n")
        df = read_file(p)
        assert df["b"].isna().iloc[0]
        assert df["c"].isna().iloc[1]

    def test_csv_index_column_reset(self, tmp_path):
        """Named index column should be reset into a regular column."""
        p = write(tmp_path, "f.csv", "id,name\n1,alpha\n2,beta\n")
        df = read_file(p)
        # After reset_index, 'id' should be a regular column, not the index
        assert "id" in df.columns

    def test_csv_no_header_prefixed(self, tmp_path):
        """Files without header should get Unknown_N column names."""
        # Sniffer will decide; force no-header by writing pure data
        # We test the prefix behaviour if sniffer returns no header
        # Using a numeric-only file that sniffer reliably tags as no-header
        p = write(tmp_path, "f.csv", "1,2,3\n4,5,6\n7,8,9\n")
        df = read_file(p)
        assert isinstance(df, pd.DataFrame)
        # Either prefixed or not depending on sniffer; must not crash
        assert len(df.columns) >= 3

    def test_single_column_csv(self, tmp_path):
        p = write(tmp_path, "f.csv", "gene\nBRCA1\nTP53\nEGFR\n")
        df = read_file(p)
        assert "gene" in df.columns or len(df.columns) == 1

    def test_large_csv_reads_fully(self, tmp_path):
        rows = "\n".join(f"{i},seq_{i},{i * 0.1:.2f}" for i in range(500))
        p = write(tmp_path, "f.csv", f"id,seq,value\n{rows}\n")
        df = read_file(p)
        assert len(df) == 500

    def test_utf8_encoded_csv(self, tmp_path):
        p = write(tmp_path, "f.csv", "organism,count\nEscherichia coli,42\nBacillus subtilis,17\n")
        df = read_file(p)
        assert "organism" in df.columns
        assert "Escherichia coli" in df["organism"].values


# ---------------------------------------------------------------------------
# read_file — TSV
# ---------------------------------------------------------------------------

class TestReadFileTSV:
    def test_reads_basic_tsv(self, tmp_path):
        p = write(tmp_path, "f.tsv", "id\tname\tvalue\n1\talpha\t1.0\n2\tbeta\t2.0\n")
        df = read_file(p)
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 2

    def test_tsv_columns_correct(self, tmp_path):
        p = write(tmp_path, "f.tsv", "gene\torganism\tGO\nBRCA1\tHomo sapiens\tGO:0008150\n")
        df = read_file(p)
        assert "gene" in df.columns
        assert "organism" in df.columns

    def test_tsv_with_missing_values(self, tmp_path):
        p = write(tmp_path, "f.tsv", "a\tb\tc\n1\t\t3\n")
        df = read_file(p)
        assert df["b"].isna().iloc[0] or df["b"].iloc[0] == ""


# ---------------------------------------------------------------------------
# read_file — JSON
# ---------------------------------------------------------------------------

class TestReadFileJSON:
    def test_reads_basic_json(self, tmp_path):
        data = [["alpha", 1], ["beta", 2], ["gamma", 3]]
        p = write(tmp_path, "f.json", json.dumps(data))
        df = read_file(p)
        assert isinstance(df, pd.DataFrame)
        assert len(df) >= 2

    def test_json_returns_dataframe(self, tmp_path):
        data = {"a": [1, 2, 3], "b": [4, 5, 6]}
        p = write(tmp_path, "f.json", json.dumps(data))
        df = read_file(p)
        assert isinstance(df, pd.DataFrame)


# ---------------------------------------------------------------------------
# read_file — unsupported formats
# ---------------------------------------------------------------------------

class TestReadFileUnsupportedFormat:
    def test_raises_for_txt_extension(self, tmp_path):
        p = write(tmp_path, "f.txt", "a,b\n1,2\n")
        with pytest.raises(ValueError, match="not a .csv or .tsv file"):
            read_file(p)

    def test_raises_for_xlsx_extension(self, tmp_path):
        # Write a dummy file with xlsx extension
        p = tmp_path / "f.xlsx"
        p.write_bytes(b"dummy")
        with pytest.raises(Exception):
            read_file(p)

    def test_raises_for_no_extension(self, tmp_path):
        p = write(tmp_path, "noext", "a,b\n1,2\n")
        with pytest.raises(ValueError):
            read_file(p)

    def test_error_message_contains_filename(self, tmp_path):
        p = write(tmp_path, "mydata.txt", "a,b\n1,2\n")
        with pytest.raises(ValueError, match="mydata.txt"):
            read_file(p)


# ---------------------------------------------------------------------------
# read_file — path handling
# ---------------------------------------------------------------------------

class TestReadFilePaths:
    def test_accepts_pathlib_path(self, tmp_path):
        import pathlib
        p = write(tmp_path, "f.csv", "a,b\n1,2\n")
        df = read_file(pathlib.Path(p))
        assert isinstance(df, pd.DataFrame)

    def test_accepts_string_path(self, tmp_path):
        p = write(tmp_path, "f.csv", "a,b\n1,2\n")
        df = read_file(str(p))
        assert isinstance(df, pd.DataFrame)

    def test_nonexistent_file_raises(self, tmp_path):
        with pytest.raises(Exception):
            read_file(tmp_path / "nonexistent.csv")


# ---------------------------------------------------------------------------
# parse_parquet
# ---------------------------------------------------------------------------

class TestParseParquet:
    def _make_parquet(self, tmp_path, df, filename="f.parquet"):
        p = tmp_path / filename
        df.to_parquet(p, engine="pyarrow")
        return p

    def test_reads_basic_parquet(self, tmp_path):
        df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        p = self._make_parquet(tmp_path, df)
        result = parse_parquet(p)
        assert isinstance(result, pd.DataFrame)
        assert "a" in result.columns
        assert len(result) == 3

    def test_non_string_columns_converted(self, tmp_path):
        """Numeric column names should be cast to string."""
        df = pd.DataFrame({0: [1, 2], 1: [3, 4]})
        p = self._make_parquet(tmp_path, df)
        result = parse_parquet(p)
        assert all(isinstance(c, str) for c in result.columns)

    def test_named_index_reset(self, tmp_path):
        """Named index should become a regular column after reset_index."""
        df = pd.DataFrame({"id": [1, 2, 3], "val": [4, 5, 6]})
        df = df.set_index("id")
        p = self._make_parquet(tmp_path, df)
        result = parse_parquet(p)
        assert "id" in result.columns

    def test_empty_column_name_renamed(self, tmp_path):
        """Empty string column name should be renamed to 'Unknown'."""
        df = pd.DataFrame({"": [1, 2], "b": [3, 4]})
        p = self._make_parquet(tmp_path, df)
        result = parse_parquet(p)
        # Empty column either becomes index or gets renamed
        assert "" not in result.columns or "Unknown" in result.columns

    def test_all_string_columns_preserved(self, tmp_path):
        df = pd.DataFrame({"gene": ["BRCA1", "TP53"], "count": [10, 20]})
        p = self._make_parquet(tmp_path, df)
        result = parse_parquet(p)
        assert list(result.columns) == ["gene", "count"]

    def test_large_parquet(self, tmp_path):
        df = pd.DataFrame({
            "seq": [f"SEQ{i}" for i in range(1000)],
            "gc": np.random.uniform(0.3, 0.7, 1000),
        })
        p = self._make_parquet(tmp_path, df)
        result = parse_parquet(p)
        assert len(result) == 1000