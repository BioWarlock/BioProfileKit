from dataclasses import dataclass

from numpy import ndarray


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
