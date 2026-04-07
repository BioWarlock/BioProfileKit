import re
from dataclasses import dataclass

import math
import numpy as np
import pandas as pd
import plotly.express as px
from numpy import ndarray
from pandas.api.types import infer_dtype
from scipy import stats

from .sequence_enum import Sequence
from .wrapper_utils import fast_check_sequence, char_entropy

def _alphabet_from_pattern(pattern):
    match = re.search(r'\[([^]]+)]', pattern.pattern)
    return set(match.group(1).upper()) if match else set()

# ToDo test if DNA and RNA Entropy is needed
DNA_ALPHABET = _alphabet_from_pattern(Sequence.DNA.value)
RNA_ALPHABET = _alphabet_from_pattern(Sequence.RNA.value)
PROTEIN_ALPHABET = _alphabet_from_pattern(Sequence.PROTEIN.value)

ENTROPY_THRESHOLDS = {
    "dna": 0.5 * math.log2(len(DNA_ALPHABET)),
    "rna": 0.5 * math.log2(len(RNA_ALPHABET)),
    "protein": 0.5 * math.log2(len(PROTEIN_ALPHABET)),
}

@dataclass
class NumericalData:
    filename: str
    rows: int
    cols: int
    nulls: int
    nulls_percentage: float
    empty_rows: int
    dup_row: int
    dup_col: int
    ratio: float
    memory: float
    alerts: int


@dataclass
class ColumnOverview:
    name: str
    number: int | None
    unique: int | None
    missing: int | None
    missing_per: float | None
    type: str
    sequence: str | None
    invalid_seqs: list[str] | None
    describe_plot: str | None
    constant: bool | None
    correlation: list[str] | None
    # taxonomy: bool


@dataclass
class NumericColumns:
    name: str
    min: float
    max: float
    mean: float
    median: float
    mode: float
    std: float
    sum: float
    kurtosis: float
    skewness: float
    coefficient_of_variation: float
    mad: float
    quantiles: ndarray
    memory: int
    value_counts: dict
    frequencies: dict

    # cardinalities: list[int]


@dataclass
class CategoricalColumns:
    name: str
    unique_categories: int
    mode: str
    entropy: float
    frequencies: dict
    gini: float
    simpson_diversity: float
    value_counts: dict
    max_category_length: int
    min_category_length: int
    memory: int
    cardinality_ratio: float


def overview(df: pd.DataFrame, file) -> NumericalData:
    return NumericalData(
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
        alerts=0
    )

#ToDo check for empty Column and return
def column_overview(df: pd.DataFrame, col) -> ColumnOverview:
    seq_type, invalid = check_sequence(df, col)
    if seq_type != "None":
        print(f"Column: {col:10s} is of type: {seq_type:10s} with invalid sequences: {invalid}.")
    return ColumnOverview(
        name=col,
        number=int(df[col].notnull().sum()),
        unique=df[col].nunique(),
        missing=int(df[col].isnull().sum()),
        missing_per=100 if df[col].isnull().all() else round(sum(df[col].isnull()) * 100 / df[col].size, 2),
        type=str(df[col].dtype),
        sequence=seq_type,
        invalid_seqs=invalid,
        describe_plot=None if df[col].isnull().all() else plot_overview(df[col]),
        constant=True if (df[col].nunique() == 1) else False,
        correlation=get_correlation(df, col),
    )

#
def numeric_columns(df: pd.DataFrame, col) -> NumericColumns:
    series = df[col].dropna()

    if series.empty:
        coefficient_of_variation = np.nan
        quantiles = np.array([np.nan, np.nan, np.nan])
        mode_value = np.nan
        value_counts = {}
        frequencies = {}
        mad_value = np.nan
        min_value = np.nan
        max_value = np.nan
        mean_value = np.nan
        median_value = np.nan
        std_value = np.nan
        sum_value = np.nan
        kurtosis_value = np.nan
        skewness_value = np.nan
    else:
        mean_value = float(series.mean())
        std_value = float(series.std())
        coefficient_of_variation = np.nan if mean_value == 0 else float(std_value / abs(mean_value))
        quantiles = np.array(series.quantile([0.25, 0.5, 0.75]).to_list(), dtype=float)
        mode_series = series.mode()
        mode_value = float(mode_series.iloc[0]) if not mode_series.empty else np.nan
        value_counts = series.value_counts().head(20).to_dict()
        frequencies = series.value_counts(normalize=True).head(20).to_dict()
        mad_value = float(stats.median_abs_deviation(series, nan_policy='omit'))
        min_value = float(series.min())
        max_value = float(series.max())
        median_value = float(series.median())
        sum_value = float(series.sum())
        kurtosis_value = float(series.kurtosis())
        skewness_value = float(series.skew())

    return NumericColumns(
        name=col,
        min=round(min_value, 2) if np.isfinite(min_value) else np.nan,
        max=round(max_value, 2) if np.isfinite(max_value) else np.nan,
        mean=round(mean_value, 2) if np.isfinite(mean_value) else np.nan,
        median=round(median_value, 2) if np.isfinite(median_value) else np.nan,
        mode=round(mode_value, 2) if np.isfinite(mode_value) else np.nan,
        std=round(std_value, 2) if np.isfinite(std_value) else np.nan,
        sum=round(sum_value, 2) if np.isfinite(sum_value) else np.nan,
        kurtosis=round(kurtosis_value, 2) if np.isfinite(kurtosis_value) else np.nan,
        skewness=round(skewness_value, 2) if np.isfinite(skewness_value) else np.nan,
        coefficient_of_variation=round(coefficient_of_variation, 2) if np.isfinite(coefficient_of_variation) else np.nan,
        mad=mad_value if np.isfinite(mad_value) else np.nan,
        quantiles=quantiles,
        memory=df[col].memory_usage(deep=True),
        value_counts=value_counts,
        frequencies=frequencies
    )


def categorical_columns(df: pd.DataFrame, col: str) -> CategoricalColumns:
    value_counts = df[col].value_counts()
    n = int(df[col].notna().sum())
    frequencies = value_counts / n if n > 0 else value_counts
    if n > 0:
        entropy = -(frequencies * np.log2(frequencies)).sum()
        gini = 1 - (frequencies ** 2).sum()
        simpson = 1 / (frequencies ** 2).sum()
        cardinality_ratio = df[col].nunique() / n
        mode_value = df[col].mode().iloc[0]
    else:
        entropy = np.nan
        gini = np.nan
        simpson = np.nan
        cardinality_ratio = np.nan
        mode_value = ""
    lengths = df[col].astype(str).str.len()

    return CategoricalColumns(
        name=col,
        unique_categories=df[col].nunique(),
        mode=mode_value,
        entropy=round(entropy, 2),
        frequencies=df[col].value_counts(normalize=True).head(20).to_dict(),
        gini=round(gini, 2),
        simpson_diversity=round(simpson, 2),
        value_counts=df[col].value_counts().head(20).to_dict(),
        max_category_length=lengths.max(),
        min_category_length=lengths.min(),
        memory=df[col].memory_usage(deep=True),
        cardinality_ratio=round(cardinality_ratio, 3)
    )


def get_correlation(df: pd.DataFrame, col) -> list | None:
    ncols = df.select_dtypes(include='number').dropna(axis=1, how='all').columns
    if col in ncols:
        std = df[ncols].std(ddof=0)
        ncols = std[std > 0].index
        if col not in ncols:
            return None
        corr = df[ncols].corrwith(df[col], method='pearson')
        corr.drop(labels=col, inplace=True)
        corr = corr[corr.abs() >= 0.3]
        if corr.empty:
            return None
        return list(zip(corr.index, corr))
    return None


# ToDo: move to plot_utils
def plot_overview(col):
    if col.dtype != 'object':
        bins = None if col.nunique() < 10 else 10
        fig = px.histogram(col, nbins=bins, color_discrete_sequence=['#0F65A0'])
        fig.update_layout(bargap=0.2, plot_bgcolor='white')
        fig.update_xaxes(
            mirror=True,
            ticks='outside',
            showline=True,
            linecolor='black',
            gridcolor='lightgrey'
        )
        fig.update_yaxes(
            mirror=True,
            ticks='outside',
            showline=True,
            linecolor='black',
            gridcolor='lightgrey'
        )
        return fig.to_html(full_html=False, include_plotlyjs=False)
    return None


# ToDo: move to sequence_utils
def check_sequence(df, col, threshold=0.95):
    if df[col].name in df.select_dtypes(include=['number', 'bool']).columns or infer_dtype(df[col]).__contains__('mixed'):
         return "None", []
    if df[col].astype(str).str.len().eq(1).all():
        return "None", []
    values = df[col].dropna().astype(str).tolist()

    #ToDo real cardinality
    unique_count = len(set(values))
    if unique_count < 10:
        return "None", []

    if all(len(x) > 2 for x in values):
        match, invalid = fast_check_sequence(values, Sequence.DNA.value, threshold)
        if match:
            return "dna", _get_invalid(values, invalid)
        match, invalid = fast_check_sequence(values, Sequence.RNA.value, threshold)
        if match:
            return "rna", _get_invalid(values, invalid)
        match, invalid = fast_check_sequence(values, Sequence.PROTEIN.value, threshold)
        if match:
            if not invalid or char_entropy(values, PROTEIN_ALPHABET) >= ENTROPY_THRESHOLDS["protein"]:
                return "protein", _get_invalid(values, invalid)
    return "None", []

def _get_invalid(values, invalid_indices):
    if not invalid_indices:
        return []
    return [values[i] for i in invalid_indices]


"""def rank_taxonomy(df, col):
    if df[col].dtype != 'object':
        return False

    results = df[col].astype(str).apply(lambda x: validate_taxonomy(x))
    results = results[~results.str.len().eq(0)]

    if not results.empty:
        results = set(deepflatten(results.value_counts().index.tolist(), depth=1))
        print(results)

    return False"""
