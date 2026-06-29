#!/usr/bin/env python3
import numpy as np
import pandas as pd
import pathlib
from functools import reduce
import csv
import click

def read_file(file: click.Path) -> pd.DataFrame | None:
    file = pathlib.Path(file.__str__()).absolute()
    ext = pathlib.Path(file.__str__()).suffix

    seperator = _get_sep(file)
    with open(file, encoding="utf-8") as csv_file:
        csv_bytes = "".join(csv_file.readline() for _ in range(1000))
        try:
            dialect = csv.Sniffer().sniff(csv_bytes)
            header = csv.Sniffer().has_header(csv_bytes)
        except csv.Error:
            # Fallback for single-column or un-sniffable files
            class DummyDialect:
                delimiter = seperator
            dialect = DummyDialect()
            header = True
        csv_file.seek(0)

    if dialect.delimiter != seperator:
        dialect.delimiter = seperator
        head_col = 0
        idx_col = None
    else:
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
        # Removed orient='values' to allow pandas to natively infer dictionaries vs arrays
        df = pd.read_json(file.__str__())

        if not df.empty:
            # Use .iloc[0] to look up positionally and avoid KeyErrors
            cols = [i for i in df.columns if isinstance(df[i].iloc[0], dict)]
            if not cols:
                df = df.T
                cols = [i for i in df.columns if isinstance(df[i].iloc[0], dict)]
                if not cols:
                    return df
                else:
                    data_frames = list()
                    for i in cols:
                        tmp = pd.json_normalize(df[i])
                        data_frames.append(tmp)
                    df = reduce(lambda left, right: pd.merge(left, right, left_index=True, right_index=True),
                                data_frames)
        return df

    return None

def _get_sep(file:pathlib.Path, candidates=(',','\t', ';','|')) -> str:
    with open(file,'r', encoding="utf-8") as f:
        samples=[l for i, l in enumerate(f) if l.strip() and i < 10]

    counts = np.array([[l.count(c) for c in candidates] for l in samples], dtype=np.uint16)
    avg_counts = counts.mean(axis=0)
    stdevs = counts.std(axis=0, ddof=1) if len(samples) > 1 else np.zeros(len(candidates))
    stdevs[avg_counts < 1] = np.inf

    return candidates[np.argmin(stdevs)]


def parse_parquet(file: click.Path) -> pd.DataFrame:
    df = pd.read_parquet(file.__str__(), engine="pyarrow")
    if df.columns[0] == ""  or df.columns[0] is None:
        df = df.set_index(df.columns[0])
    if df.index.name is not None and len(df.index.name) > 0:
        df = df.reset_index()
    if not all([isinstance(col, str) for col in df.columns]):
        df.columns = df.columns.map(str)
    if '' in df.columns or None in df.columns:
        df = df.rename(columns={"": "Unknown", None: "Unknown_None"})
    return df
