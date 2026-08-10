import numpy as np
import pandas as pd

from biological.sequence_detection import check_sequence
from models.overview import DatasetSummary, ColumnOverview
from .data_quality import check_mixed_types, check_suspect_values
from .plot_utils import plot_overview
from .multivariate_analysis import get_correlation

# Keywords that strongly suggest a taxonomy/organism column
_TAX_NAME_HINTS = frozenset({
    "organism", "species", "taxon", "taxonomy", "host", "strain",
    "genus", "family", "phylum", "class", "order", "lineage",
    "scientific_name", "scientificname", "tax", "ncbi",
})


def _check_taxonomy_candidate(df: pd.DataFrame, col: str) -> bool:
    """Lightweight heuristic: is this column likely to contain organism names?

    No network calls, no external data. Returns True when:
    - The column is a string/object dtype (numeric columns can't be organism names), AND
    - Either the column name contains a known taxonomy keyword, OR
      the values look like binomial nomenclature (two capitalised words, e.g. "Homo sapiens").

    Intentionally conservative — false negatives are better than flooding every
    string column with a "Not Checked" badge.
    """
    if not pd.api.types.is_object_dtype(df[col]):
        return False

    col_lower = col.lower().replace(" ", "_")
    if any(hint in col_lower for hint in _TAX_NAME_HINTS):
        return True

    # Value-pattern heuristic: sample up to 50 non-null values and check
    # whether a meaningful fraction looks like "Genus species [strain]"
    sample = df[col].dropna().astype(str).head(50)
    if sample.empty:
        return False

    binomial_matches = sample.str.match(
        r'^[A-Z][a-z]+ [a-z]'   # "Homo s…" or "Escherichia c…"
    ).sum()
    return binomial_matches / len(sample) >= 0.5


def _non_empty_mask(df: pd.DataFrame) -> pd.Series:
    """Rows that are not entirely null/empty-string.

    Fully empty rows are reported separately under `empty_rows`; excluding
    them here prevents them from also inflating `dup_row` and the grouped
    duplicates table, where every empty row would otherwise appear as a
    "duplicate" of every other empty row.
    """
    return ~df.replace("", np.nan).isna().all(axis=1)


def duplicate_row_groups(df: pd.DataFrame, exclude_empty: bool = True, max_indices: int = 25):
    """Group duplicate rows by content instead of listing every occurrence.

    Returns a list of dicts, sorted by occurrence count (descending):
        {
            "row": {col: value, ...},   # representative values for the group
            "count": int,               # how often this content occurs
            "indices": [int, ...],      # original row indices (capped at max_indices)
            "indices_truncated": int,   # how many indices were omitted, if any
        }

    Fully empty rows are excluded by default (see `_non_empty_mask`).
    """
    work_df = df[_non_empty_mask(df)] if exclude_empty else df
    if work_df.empty:
        return []

    dup_mask = work_df.duplicated(keep=False)
    dup_df = work_df[dup_mask]
    if dup_df.empty:
        return []

    groups = []
    group_cols = list(dup_df.columns)
    for _, group in dup_df.groupby(group_cols, dropna=False, sort=False):
        indices = group.index.tolist()
        first_row = group.iloc[0]
        row_values = {
            col: (None if pd.isna(val) else val)
            for col, val in first_row.items()
        }
        groups.append({
            "row": row_values,
            "count": len(indices),
            "indices": indices[:max_indices],
            "indices_truncated": max(0, len(indices) - max_indices),
        })

    groups.sort(key=lambda g: g["count"], reverse=True)
    return groups


def overview(df: pd.DataFrame, file) -> DatasetSummary:
    non_empty = _non_empty_mask(df)
    return DatasetSummary(
        filename=file,
        rows=df.shape[0],
        cols=df.shape[1],
        nulls=sum(df.isnull().sum()),
        nulls_percentage=100 if df.isnull().all().all() else round(sum(df.isnull().sum()) * 100 / df.size, 2),
        empty_rows=(~non_empty).sum(),
        dup_row=int(df[non_empty].duplicated().sum()),
        dup_col=int(df.columns.duplicated().sum()),
        ratio=round(df.shape[0] / df.shape[1], 3),
        memory=int(df.memory_usage(deep=True).sum()),
        alerts=0,
    )


def column_overview(df: pd.DataFrame, col) -> ColumnOverview:
    seq_type, invalid = check_sequence(df, col)
    mixed = check_mixed_types(df, col) if seq_type == "None" else None
    if mixed is not None:
        suspect = mixed[1]
    else:
        suspect = check_suspect_values(df, col)
    missing_percentage = 100 if df[col].isnull().all() else round(sum(df[col].isnull()) * 100 / df[col].size, 2)

    # Only run the candidate check for non-sequence string columns —
    # sequence columns are already classified; numeric/bool can't be taxonomy.
    taxonomy_candidate = (
        _check_taxonomy_candidate(df, col)
        if seq_type == "None"
        else False
    )

    return ColumnOverview(
        name=col,
        number=int(df[col].notnull().sum()),
        unique=df[col].nunique(),
        missing=int(df[col].isnull().sum()),
        missing_per=missing_percentage,
        density=(100 - missing_percentage),
        type=str(df[col].dtype),
        sequence=seq_type,
        invalid_seqs=invalid,
        mixed_types=mixed,
        suspect_values=suspect,
        describe_plot=None if df[col].isnull().all() else plot_overview(df[col]),
        constant=(df[col].nunique() == 1),
        correlation=get_correlation(df, col),
        cardinality_dimension_ratio=round(df[col].nunique() / len(df), 3),
        monotonicity=True if df[col].is_monotonic_increasing or df[col].is_monotonic_decreasing else False,
        taxonomy_candidate=taxonomy_candidate,
    )
