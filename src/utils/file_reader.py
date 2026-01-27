#!/usr/bin/env python3

import pandas as pd
import pathlib
from functools import reduce
import csv
import click

def read_file(file: click.Path) -> pd.DataFrame | None:
    file = pathlib.Path(file).absolute()
    ext = pathlib.Path(file.__str__()).suffix


    with open(file, encoding="utf-8") as csv_file:
        csv_bytes = "".join(csv_file.readline() for _ in range(10))
        dialect = csv.Sniffer().sniff(csv_bytes)
        header = csv.Sniffer().has_header(csv_bytes)
        csv_file.seek(0)

    head_col = 0 if header else None
    idx_col = 0 if header else None

    if not ext in (".csv", ".tsv", ".json"):
        raise ValueError(f'File {file} is not a .csv or .tsv file')

    if ext == ".csv" or ext == ".tsv":
        df = pd.read_csv(file.__str__(), header=head_col, index_col=idx_col, sep=dialect.delimiter, engine="pyarrow")
        if head_col is None:
            df = df.add_prefix("Unknown_")
        if df.index.name is not None and len(df.index.name) > 0:
            df = df.reset_index()
        return df

    elif ext == ".json":
        df = pd.read_json(file.__str__(), orient='values')
        cols = [i for i in df.columns if isinstance(df[i][0], dict)]
        if not cols:
            df = df.T
            cols = [i for i in df.columns if isinstance(df[i][0], dict)]
            if not cols:
                return df
            else:
                data_frames = list()
                for i in cols:
                    tmp = pd.json_normalize(df[i])
                    data_frames.append(tmp)
                df = reduce(lambda left, right: pd.merge(left, right, left_index=True, right_index=True), data_frames)
        return df

    return None


def parse_parquet(file: click.Path) -> pd.DataFrame:
    df = pd.read_parquet(file.__str__(), engine="pyarrow")
    if '' in df.columns or None in df.columns:
        df = df.add_prefix("Unknown_")
    if not all([isinstance(col, str) for col in df.columns]):
        df.columns = df.columns.map(str)
        df = df.add_prefix("Unknown_")
    if df.index.name is not None and len(df.index.name) > 0:
        df = df.reset_index()
    return df
