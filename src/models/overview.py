from dataclasses import dataclass
from typing import Optional


@dataclass
class DatasetSummary:
    filename: str
    rows: int
    cols: int
    nulls: int
    nulls_percentage: float
    empty_rows: int
    dup_row: int
    dup_col: int
    ratio: float
    memory: float
    alerts: int
    #ToDo Added for General Dataset Statistics Table
    n_number: int = 0
    n_dna: int = 0
    n_rna: int = 0
    n_protein: int = 0
    n_taxonomy: int = 0
    n_unit: int = 0
    n_functional: int = 0
    n_categorical: int = 0
    n_empty: int = 0


@dataclass
class ColumnOverview:
    name: str
    number: int | None
    unique: int | None
    missing: int | None
    missing_per: float | None
    density: float | None
    type: str
    sequence: str | None
    invalid_seqs: list[str] | None
    mixed_types: list[str] | None
    suspect_values: list[str] | None
    describe_plot: str | None
    constant: bool | None
    correlation: list[str] | None
    cardinality_dimension_ratio: float | None
    monotonicity: bool | None
    taxonomy: Optional = None
    measurement_data: Optional = None