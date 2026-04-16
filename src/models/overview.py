from dataclasses import dataclass


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


@dataclass
class ColumnOverview:
    name: str
    number: int | None
    unique: int | None
    missing: int | None
    missing_per: float | None
    type: str
    sequence: str | None
    invalid_seqs: list[str] | None
    mixed_types: list[str] | None
    suspect_values: list[str] | None
    describe_plot: str | None
    constant: bool | None
    correlation: list[str] | None
    cardinality_dimension_ratio: float | None
