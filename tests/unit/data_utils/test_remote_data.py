"""Tests for data_utils.remote_data.

Scope / rationale
------------------
Most functions in this module are thin wrappers around network downloads
(`requests.get`, `goatools.download_go_basic_obo`, gzip streams over HTTP
responses). Unit-testing those wrappers directly would mean either hitting
the real network (not appropriate for a unit-test suite) or mocking so much
of `requests`/`goatools` that the test would mostly just be re-asserting the
mock's own behaviour, not exercising real logic.

Instead these tests focus on the genuinely unit-testable seams:

- `_load_or_fetch`: the caching decision logic (fresh cache hit, stale
  cache, missing cache, forced refresh) - this is real, non-trivial control
  flow with no network involved once `fetch_fn` is mocked.
- `_clean_evidence_tags`, `_join`, `_clean_gene_names`, `_parse_description`:
  pure string/regex parsing functions with no I/O at all.
- `_parse_uniprot_dat`: the per-record transform, exercised by mocking
  `Bio.SwissProt.parse` to yield hand-built fake records (avoiding real
  gzip/network) instead of a real .dat.gz stream.
- `build_taxonomy`: a pure pandas transform over small, hand-built
  DataFrames, mirroring the existing `test_taxonomy.py` style.
- `get_taxonomy_raw`: the file-not-found guard and the successful read path
  (parquet I/O itself is mocked - see note below).
- The thin `get_*` wrappers (`get_gene_ontology`, `get_clusters_of_orthologous_groups`,
  `get_uniprot_trembl_metadata`, `get_uniprot_swissprot_metadata`, `get_tax_ids`):
  verified only for "do they call `_load_or_fetch` with the right cache dir /
  filename / fetch function", since their own bodies contain no other logic.

Deliberately NOT tested: `_download_gene_ontology`, `_download_cog`,
`_download_uniprot_trembl_metadata`, `_download_uniprot_swissprot_metadata`,
`_download_taxonomy` themselves - these are pure network I/O (`requests.get`,
`goatools`) with no branching logic of their own; testing them would mean
testing `requests`/`goatools`, not this project's code.
"""
import os
import time
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from data_utils.remote_data import (
    _load_or_fetch,
    _clean_evidence_tags,
    _join,
    _clean_gene_names,
    _parse_description,
    _parse_uniprot_dat,
    build_taxonomy,
    get_taxonomy_raw,
    get_gene_ontology,
    get_clusters_of_orthologous_groups,
    get_uniprot_trembl_metadata,
    get_uniprot_swissprot_metadata,
    get_tax_ids,
    GO_CACHE_DIR,
    GO_FILE,
    COG_CACHE_DIR,
    COG_FILE,
    UNIPROT_CACHE_DIR,
    UNIPROT_FILE,
    TREMBL_FILE,
    TAXONOMY_CACHE_DIR,
    TAXONOMY_VOCAB,
    TAXONOMY_FILE,
    CACHE_TIL_DAYS,
)


# ---------------------------------------------------------------------------
# _load_or_fetch
# ---------------------------------------------------------------------------
class TestLoadOrFetch:
    def test_missing_cache_calls_fetch_and_writes(self, tmp_path):
        cache_dir = tmp_path / "cache"
        fetch_fn = MagicMock(return_value=pd.DataFrame({"a": [1]}))

        with patch.object(pd.DataFrame, "to_parquet") as mock_to_parquet:
            result = _load_or_fetch(cache_dir, "file.parquet", fetch_fn, force_refresh=False)

        fetch_fn.assert_called_once()
        mock_to_parquet.assert_called_once()
        assert list(result["a"]) == [1]
        assert cache_dir.is_dir()

    def test_fresh_cache_hit_skips_fetch(self, tmp_path):
        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()
        path = cache_dir / "file.parquet"
        path.write_bytes(b"fake-parquet-bytes")

        fetch_fn = MagicMock()
        sentinel = pd.DataFrame({"cached": [True]})

        with patch("data_utils.remote_data.pd.read_parquet", return_value=sentinel) as mock_read:
            result = _load_or_fetch(cache_dir, "file.parquet", fetch_fn, force_refresh=False)

        fetch_fn.assert_not_called()
        mock_read.assert_called_once_with(path)
        assert result is sentinel

    def test_stale_cache_calls_fetch(self, tmp_path):
        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()
        path = cache_dir / "file.parquet"
        path.write_bytes(b"fake-parquet-bytes")

        stale_time = time.time() - (CACHE_TIL_DAYS + 1) * 86400
        os.utime(path, (stale_time, stale_time))

        fetch_fn = MagicMock(return_value=pd.DataFrame({"a": [1]}))

        with patch.object(pd.DataFrame, "to_parquet"):
            _load_or_fetch(cache_dir, "file.parquet", fetch_fn, force_refresh=False)

        fetch_fn.assert_called_once()

    def test_force_refresh_calls_fetch_even_if_fresh(self, tmp_path):
        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()
        path = cache_dir / "file.parquet"
        path.write_bytes(b"fake-parquet-bytes")  # freshly written -> age ~0 days

        fetch_fn = MagicMock(return_value=pd.DataFrame({"a": [1]}))

        with patch.object(pd.DataFrame, "to_parquet"), patch(
            "data_utils.remote_data.pd.read_parquet"
        ) as mock_read:
            _load_or_fetch(cache_dir, "file.parquet", fetch_fn, force_refresh=True)

        fetch_fn.assert_called_once()
        mock_read.assert_not_called()

    def test_cache_dir_is_created_if_missing(self, tmp_path):
        cache_dir = tmp_path / "nested" / "cache"
        fetch_fn = MagicMock(return_value=pd.DataFrame({"a": [1]}))

        with patch.object(pd.DataFrame, "to_parquet"):
            _load_or_fetch(cache_dir, "file.parquet", fetch_fn, force_refresh=False)

        assert cache_dir.is_dir()


# ---------------------------------------------------------------------------
# thin get_* wrappers - verify correct delegation to _load_or_fetch
# ---------------------------------------------------------------------------
class TestThinWrappers:
    def test_get_gene_ontology_delegates(self):
        with patch("data_utils.remote_data._load_or_fetch") as mock_load:
            mock_load.return_value = "sentinel"
            result = get_gene_ontology(force_refresh=True)

        assert result == "sentinel"
        args, kwargs = mock_load.call_args
        assert args[0] == GO_CACHE_DIR
        assert args[1] == GO_FILE
        assert args[3] is True

    def test_get_clusters_of_orthologous_groups_delegates(self):
        with patch("data_utils.remote_data._load_or_fetch") as mock_load:
            mock_load.return_value = "sentinel"
            result = get_clusters_of_orthologous_groups(force_refresh=False)

        assert result == "sentinel"
        args, kwargs = mock_load.call_args
        assert args[0] == COG_CACHE_DIR
        assert args[1] == COG_FILE
        assert args[3] is False

    def test_get_uniprot_trembl_metadata_delegates(self):
        with patch("data_utils.remote_data._load_or_fetch") as mock_load:
            mock_load.return_value = "sentinel"
            get_uniprot_trembl_metadata()

        args, _ = mock_load.call_args
        assert args[0] == UNIPROT_CACHE_DIR
        assert args[1] == TREMBL_FILE

    def test_get_uniprot_swissprot_metadata_delegates(self):
        with patch("data_utils.remote_data._load_or_fetch") as mock_load:
            mock_load.return_value = "sentinel"
            get_uniprot_swissprot_metadata()

        args, _ = mock_load.call_args
        assert args[0] == UNIPROT_CACHE_DIR
        assert args[1] == UNIPROT_FILE

    def test_get_tax_ids_delegates(self):
        with patch("data_utils.remote_data._load_or_fetch") as mock_load:
            mock_load.return_value = "sentinel"
            get_tax_ids(force_refresh=True)

        args, _ = mock_load.call_args
        assert args[0] == TAXONOMY_CACHE_DIR
        assert args[1] == TAXONOMY_VOCAB
        assert args[3] is True


# ---------------------------------------------------------------------------
# get_taxonomy_raw
# ---------------------------------------------------------------------------
class TestGetTaxonomyRaw:
    def test_missing_file_raises(self, tmp_path):
        with patch("data_utils.remote_data.TAXONOMY_CACHE_DIR", tmp_path):
            with pytest.raises(FileNotFoundError):
                get_taxonomy_raw()

    def test_existing_file_reads_parquet(self, tmp_path):
        path = tmp_path / TAXONOMY_FILE
        path.write_bytes(b"fake")
        sentinel = pd.DataFrame({"tax_id": [9606]})

        with patch("data_utils.remote_data.TAXONOMY_CACHE_DIR", tmp_path), patch(
            "data_utils.remote_data.pd.read_parquet", return_value=sentinel
        ) as mock_read:
            result = get_taxonomy_raw()

        mock_read.assert_called_once_with(path)
        assert result is sentinel


# ---------------------------------------------------------------------------
# _clean_evidence_tags
# ---------------------------------------------------------------------------
class TestCleanEvidenceTags:
    def test_removes_single_eco_tag(self):
        assert _clean_evidence_tags("Hemoglobin {ECO:0000255}") == "Hemoglobin"

    def test_removes_multiple_eco_tags(self):
        text = "Full=Hemoglobin {ECO:0000255|PROSITE-ProRule:PRU00238}{ECO:0000305}"
        assert _clean_evidence_tags(text) == "Full=Hemoglobin"

    def test_no_tag_returns_unchanged(self):
        assert _clean_evidence_tags("Hemoglobin") == "Hemoglobin"

    def test_none_input_returns_none(self):
        assert _clean_evidence_tags(None) is None

    def test_empty_string_returns_empty_string(self):
        assert _clean_evidence_tags("") == ""

    def test_strips_surrounding_whitespace(self):
        assert _clean_evidence_tags("Hemoglobin  {ECO:0000255}  ") == "Hemoglobin"


# ---------------------------------------------------------------------------
# _join
# ---------------------------------------------------------------------------
class TestJoin:
    def test_joins_multiple_values(self):
        assert _join(["a", "b", "c"]) == "a;b;c"

    def test_single_value(self):
        assert _join(["a"]) == "a"

    def test_empty_list_returns_none(self):
        assert _join([]) is None


# ---------------------------------------------------------------------------
# _clean_gene_names
# ---------------------------------------------------------------------------
class TestCleanGeneNames:
    def test_none_field_returns_none(self):
        assert _clean_gene_names(None) is None

    def test_empty_list_returns_none(self):
        assert _clean_gene_names([]) is None

    def test_single_gene_name(self):
        field = [{"Name": "HBB"}]
        assert _clean_gene_names(field) == "HBB"

    def test_multiple_gene_names_joined(self):
        field = [{"Name": "HBB"}, {"Name": "HBA1"}]
        assert _clean_gene_names(field) == "HBB;HBA1"

    def test_evidence_tags_stripped_from_names(self):
        field = [{"Name": "HBB {ECO:0000312}"}]
        assert _clean_gene_names(field) == "HBB"

    def test_entries_without_name_key_are_skipped(self):
        field = [{"Synonyms": ["X"]}, {"Name": "HBB"}]
        assert _clean_gene_names(field) == "HBB"

    def test_all_entries_without_name_returns_none(self):
        field = [{"Synonyms": ["X"]}]
        assert _clean_gene_names(field) is None


# ---------------------------------------------------------------------------
# _parse_description
# ---------------------------------------------------------------------------
class TestParseDescription:
    def test_none_description_returns_empty_defaults(self):
        result = _parse_description(None)
        assert result["protein_name"] is None
        assert result["alt_names"] == []

    def test_empty_string_returns_empty_defaults(self):
        result = _parse_description("")
        assert result["protein_name"] is None

    def test_recname_full_sets_protein_name(self):
        desc = "RecName: Full=Hemoglobin subunit beta;"
        result = _parse_description(desc)
        assert result["protein_name"] == "Hemoglobin subunit beta"

    def test_altname_full_appended_to_alt_names(self):
        desc = "RecName: Full=Hemoglobin subunit beta; AltName: Full=Beta-globin;"
        result = _parse_description(desc)
        assert result["protein_name"] == "Hemoglobin subunit beta"
        assert result["alt_names"] == ["Beta-globin"]

    def test_short_name_extracted(self):
        desc = "RecName: Full=Hemoglobin subunit beta; Short=HBB;"
        result = _parse_description(desc)
        assert result["short_names"] == ["HBB"]

    def test_ec_number_gets_ec_prefix(self):
        desc = "RecName: Full=Some enzyme; EC=1.1.1.1;"
        result = _parse_description(desc)
        assert result["ec_numbers"] == ["EC 1.1.1.1"]

    def test_allergen_extracted(self):
        desc = "RecName: Full=Some protein; Allergen=Ara h 1;"
        result = _parse_description(desc)
        assert result["allergens"] == ["Ara h 1"]

    def test_cd_antigen_extracted(self):
        desc = "RecName: Full=Some protein; CD_antigen=CD90;"
        result = _parse_description(desc)
        assert result["cd_antigens"] == ["CD90"]

    def test_inn_name_extracted(self):
        desc = "RecName: Full=Some protein; INN=Somename;"
        result = _parse_description(desc)
        assert result["inn_names"] == ["Somename"]

    def test_biotech_name_extracted(self):
        desc = "RecName: Full=Some protein; Biotech=Somebrand;"
        result = _parse_description(desc)
        assert result["biotech_names"] == ["Somebrand"]

    def test_evidence_tags_stripped_from_values(self):
        desc = "RecName: Full=Hemoglobin {ECO:0000255};"
        result = _parse_description(desc)
        assert result["protein_name"] == "Hemoglobin"

    def test_second_recname_full_goes_to_alt_names(self):
        # Only the *first* RecName/Full sets protein_name; anything after
        # (even another RecName block) is treated as an alternative name.
        desc = "RecName: Full=First name; RecName: Full=Second name;"
        result = _parse_description(desc)
        assert result["protein_name"] == "First name"
        assert result["alt_names"] == ["Second name"]

    def test_subname_full_goes_to_alt_names(self):
        desc = "SubName: Full=Putative protein;"
        result = _parse_description(desc)
        assert result["protein_name"] is None
        assert result["alt_names"] == ["Putative protein"]


# ---------------------------------------------------------------------------
# _parse_uniprot_dat (Bio.SwissProt.parse mocked out)
# ---------------------------------------------------------------------------
def _fake_record(**overrides):
    defaults = dict(
        accessions=["P12345"],
        entry_name="TEST_HUMAN",
        organism="Homo sapiens.",
        taxonomy_id=["9606"],
        sequence_length=146,
        description="RecName: Full=Hemoglobin subunit beta;",
        gene_name=[{"Name": "HBB"}],
    )
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


class TestParseUniprotDat:
    def _run(self, records):
        with patch("data_utils.remote_data.gzip.open") as mock_gzip_open, patch(
            "data_utils.remote_data.SwissProt.parse", return_value=iter(records)
        ):
            mock_gzip_open.return_value.__enter__ = MagicMock(return_value=MagicMock())
            mock_gzip_open.return_value.__exit__ = MagicMock(return_value=False)
            return _parse_uniprot_dat(raw_stream=MagicMock())

    def test_single_record_parsed(self):
        df = self._run([_fake_record()])
        assert len(df) == 1
        row = df.iloc[0]
        assert row["accession"] == "P12345"
        assert row["entry_name"] == "TEST_HUMAN"
        assert row["organism_name"] == "Homo sapiens"
        assert row["organism_id"] == "9606"
        assert row["length"] == 146
        assert row["protein_name"] == "Hemoglobin subunit beta"
        assert row["gene_names"] == "HBB"

    def test_record_without_accession_is_skipped(self):
        df = self._run([_fake_record(accessions=[])])
        assert df.empty

    def test_record_with_no_taxonomy_id(self):
        df = self._run([_fake_record(taxonomy_id=[])])
        assert df.iloc[0]["organism_id"] is None

    def test_record_with_no_gene_name(self):
        df = self._run([_fake_record(gene_name=[])])
        assert df.iloc[0]["gene_names"] is None

    def test_multiple_records(self):
        df = self._run([
            _fake_record(accessions=["P1"], entry_name="A_HUMAN"),
            _fake_record(accessions=["P2"], entry_name="B_HUMAN"),
        ])
        assert len(df) == 2
        assert list(df["accession"]) == ["P1", "P2"]

    def test_no_records_returns_empty_dataframe(self):
        df = self._run([])
        assert df.empty


# ---------------------------------------------------------------------------
# build_taxonomy
# ---------------------------------------------------------------------------
class TestBuildTaxonomy:
    def _base_df(self):
        return pd.DataFrame(
            [
                {"tax_id": 1, "name_txt": "Homo sapiens", "unique_name": "", "name_class": "scientific name", "parent_tax_id": 0, "rank": "species"},
                {"tax_id": 1, "name_txt": "human", "unique_name": "", "name_class": "common name", "parent_tax_id": 0, "rank": "species"},
                {"tax_id": 1, "name_txt": "man", "unique_name": "", "name_class": "genbank common name", "parent_tax_id": 0, "rank": "species"},
                {"tax_id": 2, "name_txt": "Escherichia coli", "unique_name": "", "name_class": "scientific name", "parent_tax_id": 0, "rank": "species"},
                {"tax_id": 2, "name_txt": "Escherichia coli", "unique_name": "", "name_class": "type material", "parent_tax_id": 0, "rank": "species"},
            ]
        )

    def test_scientific_name_mapped_onto_all_rows_for_taxid(self):
        result = build_taxonomy(self._base_df())
        assert set(result[result["tax_id"] == 1]["scientific_name"]) == {"Homo sapiens"}

    def test_common_and_genbank_common_names_kept(self):
        result = build_taxonomy(self._base_df())
        names = set(result[result["tax_id"] == 1]["name_txt"])
        assert {"Homo sapiens", "human", "man"} <= names

    def test_type_material_produces_type_strain_row(self):
        result = build_taxonomy(self._base_df())
        strain_rows = result[result["name_class"] == "type_strain"]
        assert len(strain_rows) == 1
        assert strain_rows.iloc[0]["name_txt"] == "Escherichia coli Escherichia coli"

    def test_result_sorted_by_tax_id(self):
        result = build_taxonomy(self._base_df())
        assert list(result["tax_id"]) == sorted(result["tax_id"])

    def test_no_type_material_yields_no_type_strain_rows(self):
        df = self._base_df()
        df = df[df["name_class"] != "type material"]
        result = build_taxonomy(df)
        assert "type_strain" not in set(result["name_class"])

    def test_duplicates_are_dropped(self):
        df = self._base_df()
        df_dup = pd.concat([df, df], ignore_index=True)
        result_dup = build_taxonomy(df_dup)
        result_orig = build_taxonomy(df)
        assert len(result_dup) == len(result_orig)

    def test_taxid_with_no_scientific_name_gets_nan(self):
        df = pd.DataFrame(
            [
                {"tax_id": 3, "name_txt": "some synonym", "unique_name": "", "name_class": "synonym", "parent_tax_id": 0, "rank": "species"},
            ]
        )
        result = build_taxonomy(df)
        assert pd.isna(result.iloc[0]["scientific_name"])

    def test_equivalent_name_class_is_kept(self):
        df = pd.DataFrame(
            [
                {"tax_id": 4, "name_txt": "Sci Name", "unique_name": "", "name_class": "scientific name", "parent_tax_id": 0, "rank": "species"},
                {"tax_id": 4, "name_txt": "Equiv Name", "unique_name": "", "name_class": "equivalent name", "parent_tax_id": 0, "rank": "species"},
            ]
        )
        result = build_taxonomy(df)
        assert "Equiv Name" in set(result["name_txt"])