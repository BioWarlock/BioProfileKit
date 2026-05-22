from dataclasses import dataclass
from typing import Optional

from numpy import ndarray

from .outliers import Outliers


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
    infinity: float
    negative_count: float
    zero_count: float
    memory: int
    value_counts: dict
    frequencies: dict
    outliers: Optional[Outliers]
