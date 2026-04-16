import numpy as np
import pandas as pd
from pandas.api.types import infer_dtype


def check_mixed_types(df, col):
    series = df[col].dropna()
    if series.empty:
        return None

    inferred = infer_dtype(series)

    if 'mixed' in inferred:
        is_numeric = series.apply(lambda x: isinstance(x, (int, float, complex)))
        numeric_count = is_numeric.sum()
        string_count = len(series) - numeric_count
        if numeric_count == 0 or string_count == 0:
            return None
        if numeric_count >= string_count:
            return series[~is_numeric].astype(str).tolist()
        else:
            return series[is_numeric].astype(str).tolist()

    if str(series.dtype) in ('str', 'string', 'object'):
        numeric_mask = pd.to_numeric(series, errors='coerce').notna() & series.notna()
        numeric_count = numeric_mask.sum()
        string_count = len(series) - numeric_count
        if numeric_count == 0 or string_count == 0:
            return None
        if numeric_count <= string_count:
            return series[numeric_mask].tolist()
        else:
            return series[~numeric_mask].tolist()

    return None


def check_suspect_values(df, col):
    series = df[col].dropna()
    if series.empty:
        return None

    dtype_str = str(series.dtype)
    if dtype_str in ('str', 'string', 'object'):
        numeric_mask = pd.to_numeric(series, errors='coerce').notna() & series.notna()
        numeric_count = numeric_mask.sum()
        if numeric_count == 0:
            return None
        if numeric_count / len(series) < 0.15:
            return series[numeric_mask].tolist()
        return None

    if pd.api.types.is_numeric_dtype(series):
        sentinels = {np.inf, -np.inf, np.nan, float('inf'), float('-inf'), float('nan')}
        found = [v for v in sentinels if v in series.values]
        if found:
            return [str(v) for v in found]
        return None

    return None