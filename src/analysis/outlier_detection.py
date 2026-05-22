from typing import Optional

import numpy as np

from cython_wrapper.medcouple_fast import fast_medcouple
from models.outliers import Outliers

def detect_outliers(values: np.ndarray) -> Optional[Outliers]:
    if len(values) < 20 or np.unique(values).size <= 10:
        return None

    q1 = np.percentile(values, 25, axis=0)
    q3 = np.percentile(values, 75, axis=0)
    iqr = q3 - q1
    mean = np.mean(values)
    std = np.std(values, ddof=1)
    median = np.median(values)
    mad = np.median(np.abs(values - median))

    if iqr <= 0:
        return None
    # Adjusted Boxplot (Hubert & Vandervieren, 2008)
    """
    Adjusted implementation adapted from robpy (v.0.0.6) 
    Source: https://robpy.readthedocs.io/en/latest/_modules/robpy/univariate/adjusted_boxplot.html
    """
    #print("Adjusted IQR")
    mc = fast_medcouple(values)

    if mc >= 0:
        lower_iqr = q1 - 1.5 * np.exp(-4 * mc) * iqr
        upper_iqr = q3 + 1.5 * np.exp(3 * mc) * iqr
    else:
        lower_iqr = q1 - 1.5 * np.exp(-3 * mc) * iqr
        upper_iqr = q3 + 1.5 * np.exp(4 * mc) * iqr

    lower_iqr = max(lower_iqr, values.min())
    upper_iqr = min(upper_iqr, values.max())

    # Modified Z-Score (MAD-based, Iglewicz & Hoaglin, 1993)
    mz_lower, mz_upper = 0, 0
    if mad > 0:
        modified_z = 0.6745 * (values - median) / mad
        mz_lower = int((modified_z < -3.5).sum())
        mz_upper = int((modified_z > 3.5).sum())

    # Standard Z-Score
    z_upper, z_lower = 0, 0
    if std > 0:
        z = (values - mean) / std
        z_lower = int((z < -3.0).sum())
        z_upper = int((z > 3.0).sum())

    return Outliers(
        lower_bound=np.round(lower_iqr, 4).astype(np.float64),
        upper_bound=np.round(upper_iqr, 4).astype(np.float64),
        n_lower_iqr=int((values < lower_iqr).sum()),
        n_upper_iqr=int((values > upper_iqr).sum()),
        medcouple=np.round(mc, 4),
        n_lower_mzscore=mz_lower,
        n_upper_mzscore=mz_upper,
        n_lower_zscore=z_lower,
        n_upper_zscore=z_upper,
    )

