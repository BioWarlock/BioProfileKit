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
    rare_categories: float
    top_5_coverage: float
    cib_ratio: float
    top_1_coverage: float
    effective_cardinality: float
    memory: int
