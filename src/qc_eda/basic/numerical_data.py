from dataclasses import dataclass

import numpy as np
import pandas as pd
import plotly.express as px
from numpy import ndarray
from pandas.api.types import infer_dtype
from scipy import stats

from .sequence_enum import Sequence
from .wrapper_utils import fast_check_sequence

"""
ToDo Numerical data:
x quantiles 
- cardinalities --> später
x constant values
- duplicated columns --> später
x correlation
x memory usage per column
- distribution <-- skewness
"""


@dataclass
class NumericalData:
    filename: str
    rows: int
    cols: int
    nulls: int
    nulls_percentage: float
    dup_row: int
    dup_col: int
    memory: float
    alerts: int


@dataclass
class ColumnOverview:
    name: str
    number: int
    unique: int
    missing: int
    missing_per: float
    type: str
    sequence: str
    describe_plot: str | None
    constant: bool
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
        nulls_percentage=round(sum(df.isnull().sum()) * 100 / df.size, 2),
        dup_row=int(df.duplicated().sum()),
        dup_col=int(df.columns.duplicated().sum()),
        memory=int(df.memory_usage(deep=True).sum()),
        alerts=0
    )


def column_overview(df: pd.DataFrame, col) -> ColumnOverview:
    return ColumnOverview(
        name=col,
        number=int(df[col].notnull().sum()),
        unique=df[col].nunique(),
        missing=int(df[col].isnull().sum()),
        missing_per=round(float(df[col].isnull().sum() * 100 / df[col].size), 2),
        type=str(df[col].dtype),
        sequence=check_sequence(df, col),
        describe_plot=plot_overview(df[col]),
        constant=True if (df[col].nunique() == 1) else False,
        correlation=get_correlation(df, col),
    )


def numeric_columns(df: pd.DataFrame, col) -> NumericColumns:
    return NumericColumns(
        name=col,
        min=round(df[col].min(), 2),
        max=round(df[col].max(), 2),
        mean=round(df[col].mean(), 2),
        median=round(df[col].median(), 2),
        mode=round(df[col].mode().iloc[0], 2),
        std=round(df[col].std(), 2),
        sum=round(df[col].sum(), 2),
        kurtosis=round(df[col].kurtosis(), 2),
        skewness=round(df[col].skew(), 2),
        mad=stats.median_abs_deviation(df[col], nan_policy='omit'),  # Ignore NaN values, set Warning
        coefficient_of_variation=round(stats.variation(df[col], nan_policy='omit'), 2),
        quantiles=stats.quantile(df[col], [0.25, 0.5, 0.75]),
        memory=df[col].memory_usage(deep=True),
        value_counts=df[col].value_counts().head(20).to_dict(),
        frequencies=df[col].value_counts(normalize=True).head(20).to_dict()

    )


def categorical_columns(df: pd.DataFrame, col: str) -> CategoricalColumns:
    value_counts = df[col].value_counts()
    n = len(df[col])
    frequencies = value_counts / n

    entropy = -(frequencies * np.log2(frequencies)).sum()
    gini = 1 - (frequencies ** 2).sum()
    simpson = 1 / (frequencies ** 2).sum()
    lengths = df[col].astype(str).str.len()

    return CategoricalColumns(
        name=col,
        unique_categories=df[col].nunique(),
        mode=df[col].mode().iloc[0],
        entropy=round(entropy, 2),
        frequencies=df[col].value_counts(normalize=True).head(20).to_dict(),
        gini=round(gini, 2),
        simpson_diversity=round(simpson, 2),
        value_counts=df[col].value_counts().head(20).to_dict(),
        max_category_length=lengths.max(),
        min_category_length=lengths.min(),
        memory=df[col].memory_usage(deep=True),
        cardinality_ratio=round(df[col].nunique() / n, 3)
    )


def get_correlation(df: pd.DataFrame, col) -> list | None:
    ncols = df.select_dtypes(include='number').columns
    if col in ncols:
        corr = df[ncols].corrwith(df[col], method='pearson')
        corr.drop(labels=col, inplace=True)
        corr = corr.drop(corr[corr < .3].index)
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
        return fig.to_html(full_html=False, include_plotlyjs='cdn')
    return None


# ToDo: move to sequence_utils
def check_sequence(df, col):
    if df[col].name in df.select_dtypes(include=['number', 'bool']).columns or infer_dtype(df[col]).__contains__('mixed'):
        return "None"
    if df[col].astype(str).str.len().eq(1).all():
        return "None"
    values = df[col].dropna().astype(str).tolist()
    if all(len(x) > 1 for x in values):
        if fast_check_sequence(values, Sequence.DNA.value):
            return "dna"
        elif fast_check_sequence(values, Sequence.RNA.value):
            return "rna"
        elif fast_check_sequence(values, Sequence.PROTEIN.value):
            return "protein"
    return "None"


"""def rank_taxonomy(df, col):
    if df[col].dtype != 'object':
        return False

    results = df[col].astype(str).apply(lambda x: validate_taxonomy(x))
    results = results[~results.str.len().eq(0)]

    if not results.empty:
        results = set(deepflatten(results.value_counts().index.tolist(), depth=1))
        print(results)

    return False"""
