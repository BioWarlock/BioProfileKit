import numpy as np
import pandas as pd
from scipy import stats

from analysis.outlier_detection import detect_outliers
from models.numeric import NumericColumns


def _safe_round(val, decimals=2):
    return round(val, decimals) if np.isfinite(val) else np.nan


def numeric_columns(df: pd.DataFrame, col) -> NumericColumns:
    series = df[col].dropna()
    print("Column: ", col)
    res = detect_outliers(series.to_numpy(dtype=np.double))

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
        outliers = None
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
        outliers = detect_outliers(series.to_numpy(dtype=np.double))

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
        infinity= series.isin([np.inf, -np.inf]).sum(),
        negative_count=np.sum((series < 0).values.ravel()),
        zero_count= np.sum(series== 0),
        memory=df[col].memory_usage(deep=True),
        value_counts=value_counts,
        frequencies=frequencies,
        outliers=outliers,
    )


