import numpy as np
import pandas as pd
from pandas.api.types import infer_dtype


def _split_numeric_string(series):
    inferred = infer_dtype(series)

    if 'mixed' in inferred:
        is_numeric = series.apply(lambda x: isinstance(x, (int, float, complex)))
        return series[is_numeric], series[~is_numeric]

    if str(series.dtype) in ('str', 'string', 'object'):
        numeric_mask = pd.to_numeric(series, errors='coerce').notna() & series.notna()
        return series[numeric_mask], series[~numeric_mask]

    return None, None


def check_mixed_types(df, col):
    series = df[col].dropna()
    if series.empty:
        return None

    numeric_part, string_part = _split_numeric_string(series)
    if numeric_part is None or len(numeric_part) == 0 or len(string_part) == 0:
        return None
    if len(numeric_part) >= len(string_part):
        return "String", string_part
    else:
        return "Numeric", numeric_part


def check_suspect_values(df, col):
    series = df[col].dropna()
    if series.empty:
        return None

    if str(series.dtype) in ('str', 'string', 'object'):
        numeric_part, _ = _split_numeric_string(series)
        if numeric_part is None or len(numeric_part) == 0:
            return None
        if len(numeric_part) / len(series) < 0.15:
            return numeric_part

    if pd.api.types.is_numeric_dtype(series):
        sentinels = {np.inf, -np.inf}
        found = [v for v in sentinels if v in series.values]
        if found:
            return pd.Series([str(v) for v in found])

    return None