import io
import os
import pathlib
import re
import sys
import zipfile
from contextlib import redirect_stdout
from pathlib import Path

import pandas as pd
import requests
from goatools.base import download_go_basic_obo
from goatools.obo_parser import GODag
import time
from pathlib import Path
from Bio import SwissProt
import gzip


if sys.platform == "win32":
    BPK_CACHE_ROOT = Path(os.environ.get("BPK_CACHE_DIR", "%LOCALAPPDAT%/bioprofilekit")).expanduser().resolve()
elif sys.platform in ["linux", "darwin"]:
    BPK_CACHE_ROOT = Path(os.environ.get("BPK_CACHE_DIR", "~/.local/share/bioprofilekit")).expanduser().resolve()
else:
    raise RuntimeError(f"Unsupported operating system: {sys.platform}")

CACHE_TIL_DAYS = 30


#CACHE_DIR = Path.cwd() / ".bioprofilekit" / "taxonomy" #ToDo change to Home --> Backend

TAXONOMY_CACHE_DIR = BPK_CACHE_ROOT / "taxonomy"
GO_CACHE_DIR = BPK_CACHE_ROOT / "go"
COG_CACHE_DIR = BPK_CACHE_ROOT / "cog"
UNIPROT_CACHE_DIR = BPK_CACHE_ROOT / "uniprot"

TAXONOMY_FILE = "taxonomy_raw.parquet"
TAXONOMY_VOCAB = "taxonomy_vocab.parquet"
GO_FILE = "go_terms.parquet"
COG_FILE = "cog_groups.parquet"
UNIPROT_FILE = "uniprot_swissport.parquet"
TREMBL_FILE = "uniprot_trembl.parquet"

SEGMENT_RE = re.compile(r'(RecName|AltName|SubName|Flags|Contains|Includes):')
FIELD_RE = re.compile(r'(Full|Short|EC|Allergen|CD_antigen|INN|Biotech)=([^;]+)')
ECO_TAG_RE = re.compile(r'\s*\{ECO:[^}]*\}')


def _load_or_fetch(cache_dir: Path, filename: str, fetch_fn, force_refresh: bool) -> pd.DataFrame:
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / filename
    if not force_refresh and path.is_file():
        age_days = (time.time() - path.stat().st_mtime) / 86400
        if age_days < CACHE_TIL_DAYS:
            return pd.read_parquet(path)

    df = fetch_fn()
    df.to_parquet(path, index=False)
    return df

def get_gene_ontology(force_refresh: bool = False) -> pd.DataFrame:
    return _load_or_fetch(GO_CACHE_DIR, GO_FILE, _download_gene_ontology, force_refresh)

def _download_gene_ontology():
    with redirect_stdout(io.StringIO()):
        obo_path = download_go_basic_obo()
    go_dag = GODag(obo_path, prt=None)

    data = [[go_id, term.name, getattr(term, "namespace", "")] for go_id, term in go_dag.items()]

    df = pd.DataFrame(data, columns=["GO_ID", "Name", "Namespace"])
    if Path(obo_path).is_file():
        pathlib.Path(obo_path).unlink(missing_ok=True)
    return df


def get_clusters_of_orthologous_groups(force_refresh: bool = False) -> pd.DataFrame:
    return _load_or_fetch(COG_CACHE_DIR, COG_FILE, _download_cog, force_refresh)

def _download_cog() -> pd.DataFrame:
    url: str = "https://ftp.ncbi.nlm.nih.gov/pub/COG/COG2024/data/cog-24.def.tab"
    fields = ["COG_ID", "Functional Category", "COG name"]

    response = requests.get(url, timeout=60)
    response.raise_for_status()
    return pd.read_csv(io.StringIO(response.text), sep="\t", skipinitialspace=True, usecols=[0, 1, 2], names=fields,)

def get_uniprot_trembl_metadata(force_refresh: bool = False) -> pd.DataFrame:
    # ToDo: add division
    return _load_or_fetch(UNIPROT_CACHE_DIR, TREMBL_FILE, _download_uniprot_trembl_metadata, force_refresh)

def _download_uniprot_trembl_metadata() -> pd.DataFrame:
    #ToDo add division
    url = "https://ftp.uniprot.org/pub/databases/uniprot/current_release/knowledgebase/complete/uniprot_trembl.dat.gz"
    print("Downloading UniProt TrEMBL flat file (this is very large, may take a long time) ...")
    response = requests.get(url, stream=True, timeout=600)
    response.raise_for_status()
    return _parse_uniprot_dat(response.raw)

def get_uniprot_swissprot_metadata(force_refresh: bool = False) -> pd.DataFrame:
    return _load_or_fetch(UNIPROT_CACHE_DIR, UNIPROT_FILE, _download_uniprot_swissprot_metadata, force_refresh)

def _download_uniprot_swissprot_metadata() -> pd.DataFrame:
    url = "https://ftp.uniprot.org/pub/databases/uniprot/current_release/knowledgebase/complete/uniprot_sprot.dat.gz"
    # print("Downloading UniProt Swiss-Prot flat file ...")
    response = requests.get(url, stream=True, timeout=600)
    response.raise_for_status()

    return _parse_uniprot_dat(response.raw)

def _parse_uniprot_dat(raw_stream) -> pd.DataFrame:
    records = []
    with gzip.open(raw_stream, "rt") as fh:
        for record in SwissProt.parse(fh):
            accession = record.accessions[0] if record.accessions else None
            if accession is None:
                continue

            parsed = _parse_description(record.description)

            records.append({
                "accession": accession,
                "entry_name": record.entry_name,
                "organism_name": record.organism.rstrip("."),
                "organism_id": record.taxonomy_id[0] if record.taxonomy_id else None,
                "length": record.sequence_length,
                "protein_name": parsed["protein_name"],
                "alt_names": _join(parsed["alt_names"]),
                "short_names": _join(parsed["short_names"]),
                "ec_numbers": _join(parsed["ec_numbers"]),
                "gene_names": _clean_gene_names(record.gene_name),
                "allergens": _join(parsed["allergens"]),
                "cd_antigens": _join(parsed["cd_antigens"]),
                "inn_names": _join(parsed["inn_names"]),
                "biotech_names": _join(parsed["biotech_names"]),
            })

    return pd.DataFrame(records)

def _clean_evidence_tags(text: str) -> str:
    return ECO_TAG_RE.sub('', text).strip() if text else text


def _join(values: list) -> str | None:
    return ";".join(values) if values else None


def _clean_gene_names(gene_name_field) -> str | None:
    if not gene_name_field:
        return None
    names = [_clean_evidence_tags(g.get("Name", "")) for g in gene_name_field if g.get("Name")]
    return ";".join(names) if names else None

def _parse_description(description: str) -> dict:
    result = {
        "protein_name": None,
        "alt_names": [], "short_names": [], "ec_numbers": [],
        "allergens": [], "cd_antigens": [], "inn_names": [], "biotech_names": [],
    }
    if not description:
        return result

    parts = SEGMENT_RE.split(description)
    segments = [(parts[i], parts[i + 1]) for i in range(1, len(parts) - 1, 2)]

    for tag, content in segments:
        for field_name, value in FIELD_RE.findall(content):
            value = _clean_evidence_tags(value)
            if not value:
                continue
            if field_name == "Full":
                if tag == "RecName" and result["protein_name"] is None:
                    result["protein_name"] = value
                else:
                    result["alt_names"].append(value)
            elif field_name == "Short":
                result["short_names"].append(value)
            elif field_name == "EC":
                result["ec_numbers"].append(f"EC {value}")
            elif field_name == "Allergen":
                result["allergens"].append(value)
            elif field_name == "CD_antigen":
                result["cd_antigens"].append(value)
            elif field_name == "INN":
                result["inn_names"].append(value)
            elif field_name == "Biotech":
                result["biotech_names"].append(value)

    return result

def get_tax_ids(force_refresh: bool = False):
    return _load_or_fetch(TAXONOMY_CACHE_DIR, TAXONOMY_VOCAB, _build_taxonomy_vocab, force_refresh)

def get_taxonomy_raw() -> pd.DataFrame:
    path = TAXONOMY_CACHE_DIR / TAXONOMY_FILE
    if not path.is_file():
        raise FileNotFoundError(
            f"{path} not found - run get_tax_ids() first to download and cache the taxonomy data."
        )
    return pd.read_parquet(path)


def _build_taxonomy_vocab() -> pd.DataFrame:
    raw = _download_taxonomy()
    raw.to_parquet(TAXONOMY_CACHE_DIR / TAXONOMY_FILE, index=False)
    return build_taxonomy(raw)


def _download_taxonomy():
    url = "https://ftp.ncbi.nih.gov/pub/taxonomy/taxdmp.zip"

    #print(f"Downloading {url} ...")
    resp = requests.get(url, stream=True)
    resp.raise_for_status()

    with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
        #print("Files inside ZIP:", zf.namelist())

        with zf.open("names.dmp") as fh:
            #print(fh)
            names = pd.read_csv(
                fh,
                sep="|",
                header=None,
                index_col=False,
                names=["tax_id", "name_txt", "unique_name", "name_class"],
                engine="c"
            )

        with zf.open("nodes.dmp") as fh:
            nodes = pd.read_csv(fh, sep="|", header=None, index_col=False, usecols=[0, 1, 2], names=["tax_id", "parent_tax_id", "rank"], engine="c")

    names = names.map(lambda x: x.strip() if isinstance(x, str) else x)
    nodes = nodes.map(lambda x: x.strip() if isinstance(x, str) else x)
    df = names.merge(nodes, on="tax_id", how="left")

    return df

def build_taxonomy(tax_df: pd.DataFrame) -> pd.DataFrame:
    name_classes = ['scientific name', 'synonym', 'equivalent name','genbank common name', 'common name']
    sci = (tax_df[tax_df['name_class'] == 'scientific name'].drop_duplicates('tax_id').set_index('tax_id')['name_txt'])
    names = tax_df[tax_df['name_class'].isin(name_classes)][['tax_id', 'name_txt', 'name_class', 'rank']].copy()
    strains = tax_df[tax_df['name_class'] == 'type material'][['tax_id', 'name_txt']]
    strains = strains.rename(columns={'name_txt': 'strain'})

    combos = names.merge(strains, on="tax_id", how="inner")
    combos['name_txt'] = combos['name_txt'] + ' ' + combos['strain']
    combos['name_class']= 'type_strain'
    combos = combos[['tax_id', 'name_txt', 'name_class', 'rank']]

    vocab = pd.concat([names, combos], ignore_index=True)
    vocab['scientific_name'] = vocab['tax_id'].map(sci)
    vocab = vocab.drop_duplicates().reset_index(drop=True)
    return vocab.sort_values(by=['tax_id'], ascending=True)
