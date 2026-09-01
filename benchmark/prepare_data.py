import random
import json

import numpy as np
import pandas as pd
import string
from datetime import datetime


def main():
    log = []
    file = "iedb"

    df = pd.read_csv("../data/iedb.tsv", sep="\t", index_col=0, low_memory=False)
    df = create_subset(df, keep_perc=0.75, seed=42, file=file, log=log)
    df = inject_numeric_in_text(df, seed=42, log=log)
    df = inject_invalid_chars(df, 10, seed=42, log=log)
    df = inject_sequence_anomalies(df, "peptide", n_affected=20, seed=42, mode="random_ascii", log=log)
    df = duplicate_column(df, 42, log=log)
    df = duplicate_rows(df, 42, log=log)
    df = empty(df, seed=42, log=log)


    summarize_log(log, file)
    df.to_csv("../data/iedb_subset_75.tsv", sep="\t", index=True, header=True)
    return df


def create_subset(df: pd.DataFrame, keep_perc: float, seed: int, file, log: list):
    n_keep = int(len(df) * keep_perc)
    df_ = df.sample(n=n_keep, random_state=seed)
    removed = df.drop(df_.index)

    log_entry = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "action": "create_subset",
        "rows": len(df),
        "columns": len(df.columns),
        "percentage": keep_perc,
        "kept_rows": len(df_),
        "kept_columns": len(df_.columns),
        "removed_datapoints": len(removed),
        "seed": seed,
        "removed_indices": removed.index.tolist(),
    }
    #removed.to_csv(f"removed_datapoints_{file}.tsv", sep="\t")
    log.append(log_entry)
    return df_


def duplicate_column(df: pd.DataFrame, seed: int, log: list):
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
    log.append(log_entry)
    return df


def duplicate_rows(df: pd.DataFrame, seed: int, log: list):
    dups = df.duplicated().sum()
    rng = random.Random(seed)
    n_dup = 100
    dup_indices = rng.sample(df.index.tolist(), n_dup)

    dup_rows = df.loc[dup_indices]
    df_new = pd.concat([df, dup_rows], ignore_index=True)

    log_entry = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "action": "duplicate_rows",
        "orginal_dups": dups,
        "dup_rows": len(dup_rows),
        "n_duplicated": n_dup,
        "duplicated_indices": dup_indices,
        "sum_duplicated": df_new.duplicated().sum(),
        "seed": seed,
    }
    log.append(log_entry)
    return df_new


def empty(df: pd.DataFrame, seed: int, log: list):
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
        "delete_rows": n_empty,
        "n_rows_emptied": len(emptied_indices),
        "empty_col": col_name,
        "emptied_indices": emptied_indices,
        "seed": seed,
    }
    log.append(log_entry)
    return df_new


def inject_numeric_in_text(df: pd.DataFrame, seed: int, log: list):
    rng = random.Random(seed)
    col = rng.choice(df.select_dtypes(exclude=["float64", "int64"]).columns.tolist())
    df_new = df.copy()

    valid_idx = df_new[df_new[col].notna()].index.tolist()
    n_affected = 50
    affected_indices = rng.sample(valid_idx, n_affected)

    original_values = {}
    for idx in affected_indices:
        val = str(df_new.at[idx, col])
        original_values[idx] = val
        if len(val) == 0:
            continue
        pos = rng.randrange(len(val))
        digit = rng.choice(string.digits)
        df_new.at[idx, col] = val[:pos] + digit + val[pos + 1:]
    log_entry = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "action": "inject_numeric_in_text",
        "column": col,
        "n_affected": n_affected,
        "affected_indices": affected_indices,
        "original_values": original_values,
        "seed": seed,
    }
    log.append(log_entry)
    return df_new


def inject_invalid_chars(df: pd.DataFrame, n_affected: int, seed: int, log: list):
    rng = random.Random(seed)
    df_new = df.copy()
    invalid_chars = ["#", "@", "$", "%", "&", "\x00", "\x1f", "§"]

    all_cells = [
        (idx, col)
        for col in df_new.columns
        for idx in df_new[df_new[col].notna()].index
    ]
    affected_cells = rng.sample(all_cells, min(n_affected, len(all_cells)))

    original_values = {}
    anomaly_types = {}

    for idx, col in affected_cells:
        val = df_new.at[idx, col]
        original_values[(idx, col)] = val

        if pd.api.types.is_numeric_dtype(df_new[col]):
            df_new.at[idx, col] = rng.choice([np.inf, -np.inf])
            anomaly_types[(idx, col)] = "infinite"
        else:
            str_val = str(val)
            if len(str_val) == 0:
                continue
            pos = rng.randrange(len(str_val))
            char = rng.choice(invalid_chars)
            df_new.at[idx, col] = str_val[:pos] + char + str_val[pos + 1:]
            anomaly_types[(idx, col)] = "invalid_char"

    log_entry = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "action": "inject_invalid_chars",
        "n_affected": len(affected_cells),
        "affected_cells": [f"{idx}|{col}" for idx, col in affected_cells],
        "anomaly_types": {f"{idx}|{col}": t for (idx, col), t in anomaly_types.items()},
        "original_values": {f"{idx}|{col}": v for (idx, col), v in original_values.items()},
        "seed": seed,
    }
    log.append(log_entry)
    return df_new


def inject_sequence_anomalies(df: pd.DataFrame, column: str, n_affected: int, seed: int, log: list,
                               mode: str = "random_ascii"):
    dna = str.maketrans("ACGTacgt", "TGCAtgca")
    rng = random.Random(seed)
    df_new = df.copy()

    valid_idx = df_new[df_new[column].notna()].index.tolist()
    affected_indices = rng.sample(valid_idx, min(n_affected, len(valid_idx)))

    original_values = {}
    for idx in affected_indices:
        val = str(df_new.at[idx, column])
        original_values[idx] = val
        if len(val) == 0:
            continue

        if mode == "random_ascii":
            pos = rng.randrange(len(val))
            invalid_char = rng.choice("!\"#$%&/()=?*+~0123456789")
            df_new.at[idx, column] = val[:pos] + invalid_char + val[pos + 1:]

        elif mode == "reverse_complement":
            df_new.at[idx, column] = val.translate(dna)[::-1]

    log_entry = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "action": "inject_sequence_anomalies",
        "column": column,
        "mode": mode,
        "n_affected": len(affected_indices),
        "affected_indices": affected_indices,
        "original_values": original_values,
        "seed": seed,
    }
    log.append(log_entry)
    return df_new


def summarize_log(log: list[dict], file: str):
    with open(f"log_{file}.json", "w") as f:
        json.dump(log, f, indent=2, default=str)

    print(f"\n{'=' * 70}")
    print(f"Manipulation Summary: {file}")
    print(f"{'=' * 70}")
    print(f"{'Step':<3} {'Action':<26} {'n_affected':<12} {'Details'}")
    print(f"{'-' * 70}")

    for i, entry in enumerate(log, start=1):
        action = entry.get("action", "-")

        n_affected = (
            entry.get("n_affected")
            or entry.get("kept_rows")
            or entry.get("n_rows_emptied")
            or entry.get("n_duplicated")
            or "-"
        )

        details = (
            entry.get("column")
            or entry.get("original_column")
            or entry.get("mode")
            or entry.get("empty_col")
            or ""
        )

        print(f"{i:<3} {action:<26} {str(n_affected):<12} {details}")

    print(f"{'-' * 70}")
    print(f"Total steps: {len(log)}  ->  log_{file}.json\n")


if __name__ == "__main__":
    main()