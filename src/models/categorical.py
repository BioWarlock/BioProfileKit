from dataclasses import dataclass


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
    cardinality_ratio: float
    memory: int
