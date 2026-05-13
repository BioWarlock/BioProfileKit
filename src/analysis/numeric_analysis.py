import math
import numpy as np
import pandas as pd
from scipy import stats

from cython_wrapper.medcouple_fast import fast_medcouple
from models.numeric import NumericColumns
from statsmodels.stats.stattools import medcouple

def _safe_round(val, decimals=2):
    return round(val, decimals) if np.isfinite(val) else np.nan


def numeric_columns(df: pd.DataFrame, col) -> NumericColumns:
    series = df[col].dropna()
    print("Column: ", col)
    #ToDo: Check outlier detection
    #detect_outliers(series)
    #print(outliers_upper)
    #iqr_t = stats.iqr(series,nan_policy='omit')
    #print(f"Compare IQR {col} {math.isclose(iqr,iqr_t, rel_tol=1e-9)}")

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
        infinity= series.isin([np.inf, -np.inf]).sum(),
        negative_count=np.sum((series < 0).values.ravel()),
        zero_count= np.sum(series== 0),
        memory=df[col].memory_usage(deep=True),
        value_counts=value_counts,
        frequencies=frequencies
    )

def detect_outliers(series):
    if series.nunique() <= 10 or len(series) < 20:
        return None

    values = series.to_numpy()
    skew = series.skew()
    q1 = np.percentile(values, 25, axis=0)
    q3 = np.percentile(values, 75, axis=0)
    iqr = q3 - q1
    mean = series.mean()
    std = series.std()
    median = np.median(values)
    mad = np.median(np.abs(values - median))

    results = {}

    # 1. Adjusted Boxplot (Hubert & Vandervieren, 2008)
    """
    Adjusted implementation adapted from robpy (v.0.0.6) 
    Source: https://robpy.readthedocs.io/en/latest/_modules/robpy/univariate/adjusted_boxplot.html
    """
    #print("Adjusted IQR")
    if iqr > 0:
        mc = medcouple(values, axis=0)
        fmc = fast_medcouple(values)
        print(f"Medcouple {mc} vs {fmc}")
        #print(mc)
        if mc >= 0:
            lower_iqr = q1 - 1.5 * np.exp(-4 * mc) * iqr
            upper_iqr = q3 + 1.5 * np.exp(3 * mc) * iqr
        else:
            lower_iqr = q1 - 1.5 * np.exp(-3 * mc) * iqr
            upper_iqr = q3 + 1.5 * np.exp(4 * mc) * iqr

        # Clamp to data range
        lower_iqr = max(lower_iqr, values.min())
        upper_iqr = min(upper_iqr, values.max())

        results["iqr"] = {
            "method": "Adjusted Boxplot (Hubert & Vandervieren, 2008)",
            "medcouple": np.round(mc, 4),
            "bounds": (np.round(lower_iqr, 4), np.round(upper_iqr, 4)),
            "n_upper": int((values > upper_iqr).sum()),
            "n_lower": int((values < lower_iqr).sum()),
        }

    # 2. Modified Z-Score (MAD-based, Iglewicz & Hoaglin, 1993)
    #print("MAD-based Z-Score")
    if mad > 0:
        modified_z = 0.6745 * (values - median) / mad
        results["modified_zscore"] = {
            "method": "Modified Z-Score (MAD)",
            "threshold": 3.5,
            "n_upper": int((modified_z > 3.5).sum()),
            "n_lower": int((modified_z < -3.5).sum()),
        }

    # 3. Standard Z-Score
    #print("Z-Score")
    if std > 0:
        z = (values - mean) / std
        results["zscore"] = {
            "method": "Z-Score",
            "threshold": 3.0,
            "n_upper": int((z > 3.0).sum()),
            "n_lower": int((z < -3.0).sum()),
        }

    if not results:
        return None

    results["summary"] = {
        "skewness": round(skew, 2),
        "iqr": np.round(iqr, 4),
        "medcouple": np.round(mc, 4) if iqr > 0 else None,
        "mad": np.round(mad, 4),
        "std": np.round(std, 4),
    }

    return results