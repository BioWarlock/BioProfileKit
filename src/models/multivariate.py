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
    mutual_information_plot: str | None
    mi_relationship_plots: list | None

    # Computed metrics
    correlation_matrix: pd.DataFrame  # combined, absolute values
    correlation_methods: pd.DataFrame  # method used per pair
    top_associations: list | None

    feature_target_correlation: dict | None  # needs target
    feature_target_plot: str | None   # needs target
    mutual_information: dict | None  # needs target
    mcar_result: dict | None  # ToDo replace with finale Littles MCAR
