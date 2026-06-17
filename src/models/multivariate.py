from dataclasses import dataclass
from typing import Optional

import pandas as pd

@dataclass
class MultivariateAnalysis:
    # Plots (HTML strings)
    correlation_heatmap: str
    pearson_heatmap: Optional[str]  # nur numerisch, -1..1
    cramers_heatmap: Optional[str]  # nur kategorisch, 0..1
    eta_heatmap: Optional[str]
    missing_matrix: str
    missing_values_barchart: str
    balance_plot: str | None
    boxplot: str
    scatter_matrix: str

    # Computed metrics
    correlation_matrix: pd.DataFrame  # combined, absolute values
    correlation_methods: pd.DataFrame  # method used per pair
    top_associations: list | None
    feature_target_correlation: dict | None  # needs target
    mutual_information: dict | None  # needs target
    mcar_result: dict | None  # replace with finale Littles MCAR
