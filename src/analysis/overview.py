import numpy as np
import pandas as pd

from biological.sequence_detection import check_sequence
from models.overview import DatasetSummary, ColumnOverview
from .data_quality import check_mixed_types, check_suspect_values
from .plot_utils import plot_overview
from .multivariate import get_correlation


def overview(df: pd.DataFrame, file) -> DatasetSummary:
    return DatasetSummary(
        filename=file,
        rows=df.shape[0],
        cols=df.shape[1],
        nulls=sum(df.isnull().sum()),
        nulls_percentage=100 if df.isnull().all().all() else round(sum(df.isnull().sum()) * 100 / df.size, 2),
        empty_rows=df.replace("", np.nan).isna().all(axis=1).sum(),
        dup_row=int(df.duplicated().sum()),
        dup_col=int(df.columns.duplicated().sum()),
        ratio=round(df.shape[0] / df.shape[1], 3),
        memory=int(df.memory_usage(deep=True).sum()),
        alerts=0,
    )


def column_overview(df: pd.DataFrame, col) -> ColumnOverview:
    seq_type, invalid = check_sequence(df, col)
    if seq_type != "None":
        print(f"Column: {col:10s} is of type: {seq_type:10s} with invalid sequences: {invalid}.")
    mixed = check_mixed_types(df, col) if seq_type == "None" else None
    if mixed is not None:
        suspect = mixed[1]
    else:
        suspect = check_suspect_values(df, col)
    missing_percentage = 100 if df[col].isnull().all() else round(sum(df[col].isnull()) * 100 / df[col].size, 2)
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
    )