from dataclasses import dataclass

import numpy as np


@dataclass
class Outliers:
    lower_bound: np.float64
    upper_bound: np.float64
    n_lower_iqr: int
    n_upper_iqr: int
    medcouple: np.float64
    n_lower_mzscore: int
    n_upper_mzscore: int
    n_lower_zscore: int
    n_upper_zscore: int