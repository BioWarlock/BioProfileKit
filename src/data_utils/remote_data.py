import io
import pathlib
import zipfile
from pathlib import Path

import pandas as pd
import requests
from goatools.base import download_go_basic_obo
from goatools.obo_parser import GODag

def get_gene_ontology():
    obo_path = download_go_basic_obo()
    go_dag = GODag(obo_path)
    print(go_dag.version)
    data = list()
    for go_id in go_dag.keys():
        term = go_dag[go_id]
        namespace = getattr(term, "namespace", "")
        data.append([go_id, term.name, namespace])

    df = pd.DataFrame(data, columns=["GO_ID", "Name", "Namespace"])
    if Path(obo_path).is_file():
        pathlib.Path(obo_path).unlink(missing_ok=True)
        print(f"Removed {obo_path}")
    return df


def get_clusters_of_orthologous_groups():
    url: str = "https://ftp.ncbi.nlm.nih.gov/pub/COG/COG2024/data/cog-24.def.tab"
    fields = ["COG_ID", "Functional Category", "COG name"]
    response = requests.get(url)

    if response.status_code == 200:
        df = pd.read_csv(io.StringIO(response.text), sep="\t", skipinitialspace=True, usecols=[0, 1, 2], names=fields)
    else:
        print(f"Error: {response.status_code}")
    return df


def get_tax_ids():
    url = "https://ftp.ncbi.nih.gov/pub/taxonomy/taxdmp.zip"

    print(f"Downloading {url} ...")
    resp = requests.get(url, stream=True)
    resp.raise_for_status()

    with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
        print("Files inside ZIP:", zf.namelist())

        with zf.open("names.dmp") as fh:
            print(fh)
            df = pd.read_csv(
                fh,
                sep="|",
                header=None,
                index_col=False,
                names=["tax_id", "name_txt", "unique_name", "name_class"],
                engine="c"
            )
    df = df.map(lambda x: x.strip() if isinstance(x, str) else x)
    #df = df[df["name_class"] == "scientific name"]
    return df

