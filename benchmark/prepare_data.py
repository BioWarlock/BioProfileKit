import random

import numpy as np
import pandas as pd

from datetime import datetime


def main():
    df = pd.read_csv("../data/iedb.tsv", sep="\t", index_col=0, low_memory=False)
    df = create_subset(df, keep_perc=0.75, seed=42, file="iedb")
    df = duplicate_column(df, 42)
    df = duplicate_rows(df,  42)
    df = empty(df, seed=42)
    print(df)


def create_subset(df: pd.DataFrame, keep_perc: float, seed: int, file):
    n_keep = int(len(df) * keep_perc)
    df_ = df.sample(n=n_keep, random_state=seed)
    removed = df.drop(df_.index)

    log_entry = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "rows": len(df),
        "columns": len(df.columns),
        "percentage": keep_perc,
        "kept_rows": len(df_),
        "kept_columns": len(df_.columns),
        "removed_datapoints": len(removed),
        "seed": seed,
        #"removed_indces": removed.index.tolist()
    }
    #removed.to_csv(f"removed_datapoints_{file}.tsv", sep="\t")
    print(log_entry)
    return df_

def duplicate_column(df: pd.DataFrame, seed:int):
    rng = random.Random(seed)
    col = rng.choice(df.columns.tolist())
    dup_name = f"{col}_dup"
    df[dup_name] = df[col]

    log_entry = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "action": "duplicate_column",
        "original_column": col,
        "dup_name": dup_name,
        "seed": seed,
    }
    print(log_entry)
    return df

def duplicate_rows(df: pd.DataFrame, seed: int):
    dups = df.duplicated().sum()
    rng = random.Random(seed)
    n_dup = 100
    dup_indices = rng.sample(df.index.tolist(), 100)

    dup_rows = df.loc[dup_indices]
    df_new = pd.concat([df, dup_rows], ignore_index=True)

    log_entry = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "action": "duplicate_rows",
        "orginal_dups": dups,
        "dup_rows": len(dup_rows),
        "n_duplicated": n_dup,
        "duplicated_indices": len(dup_indices),
        "sum_duplicated": df_new.duplicated().sum(),
        "seed": seed,
    }
    print(log_entry)
    return df_new


def empty(df: pd.DataFrame, seed: int):
    rng = random.Random(seed)
    df_new = df.copy()

    n_empty = 50
    emptied_indices = rng.sample(df_new.index.tolist(), n_empty)
    df_new.loc[emptied_indices] = np.nan

    col_name = "empty_col"
    df_new[col_name] = np.nan

    log_entry = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "action": "empty",
        "delete_rows": 50,
        "n_rows_emptied": len(emptied_indices),
        "empty_col": col_name,
        #"emptied_indices": emptied_indices,
        "seed": seed,
    }
    print(log_entry)
    return df_new


if __name__ == "__main__":
    main()