import io
import os
import pathlib
import zipfile
from pathlib import Path

import pandas as pd
import requests
from goatools.base import download_go_basic_obo
from goatools.obo_parser import GODag
import time
from pathlib import Path

BPK_CACHE_ROOT = Path(os.environ.get("BPK_CACHE_DIR", Path.cwd() / ".bioprofilekit"))
CACHE_TIL_DAYS = 30


#CACHE_DIR = Path.cwd() / ".bioprofilekit" / "taxonomy" #ToDo change to Home --> Backend

TAXONOMY_CACHE_DIR = BPK_CACHE_ROOT / "taxonomy"
GO_CACHE_DIR = BPK_CACHE_ROOT / "go"
COG_CACHE_DIR = BPK_CACHE_ROOT / "cog"

TAXONOMY_FILE = "taxonomy_raw.parquet"
TAXONOMY_VOCAB = "taxonomy_vocab.parquet"
GO_FILE = "go_terms.parquet"
COG_FILE = "cog_groups.parquet"

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
    obo_path = download_go_basic_obo()
    go_dag = GODag(obo_path)

    data = [[go_id, term.name, getattr(term, "namespace", "")] for go_id, term in go_dag.items()]

    df = pd.DataFrame(data, columns=["GO_ID", "Name", "Namespace"])
    if Path(obo_path).is_file():
        pathlib.Path(obo_path).unlink(missing_ok=True)
        print(f"Removed {obo_path}")
    return df


def get_clusters_of_orthologous_groups(force_refresh: bool = False) -> pd.DataFrame:
    return _load_or_fetch(COG_CACHE_DIR, COG_FILE, _download_cog, force_refresh)

def _download_cog() -> pd.DataFrame:
    url: str = "https://ftp.ncbi.nlm.nih.gov/pub/COG/COG2024/data/cog-24.def.tab"
    fields = ["COG_ID", "Functional Category", "COG name"]

    response = requests.get(url, timeout=60)
    response.raise_for_status()
    return pd.read_csv(io.StringIO(response.text), sep="\t", skipinitialspace=True, usecols=[0, 1, 2], names=fields,)
    """if response.status_code == 200:
        df = pd.read_csv(io.StringIO(response.text), sep="\t", skipinitialspace=True, usecols=[0, 1, 2], names=fields)
    else:
        print(f"Error: {response.status_code}")
    return df"""


def get_tax_ids(force_refresh: bool = False):
    return _load_or_fetch(TAXONOMY_CACHE_DIR, TAXONOMY_VOCAB, _build_taxonomy_vocab, force_refresh)

def _build_taxonomy_vocab() -> pd.DataFrame:
    raw = _download_taxonomy()
    raw.to_parquet(TAXONOMY_CACHE_DIR / TAXONOMY_FILE, index=False)
    return build_taxonomy(raw)


def _download_taxonomy():
    url = "https://ftp.ncbi.nih.gov/pub/taxonomy/taxdmp.zip"

    print(f"Downloading {url} ...")
    resp = requests.get(url, stream=True)
    resp.raise_for_status()

    with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
        print("Files inside ZIP:", zf.namelist())

        with zf.open("names.dmp") as fh:
            print(fh)
            names = pd.read_csv(
                fh,
                sep="|",
                header=None,
                index_col=False,
                names=["tax_id", "name_txt", "unique_name", "name_class"],
                engine="c"
            )

        with zf.open("nodes.dmp") as fh:
            nodes = pd.read_csv(fh, sep="|", header=None, index_col=False, usecols=[0,2], names=["tax_id", "rank"], engine="c")

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
