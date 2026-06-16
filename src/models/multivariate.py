from dataclasses import dataclass


@dataclass
class MultivariateAnalysis:
    correlation_heatmap: str
    missing_matrix: str
    missing_values_barchart: str
    balance_plot: str | None
    boxplot: str
    scatter_matrix: str
