import numpy as np
import pandas as pd

from models.categorical import CategoricalColumns


def categorical_columns(df: pd.DataFrame, col: str) -> CategoricalColumns:
    value_counts = df[col].value_counts()
    n = int(df[col].notna().sum())
    frequencies = value_counts / n if n > 0 else value_counts
    rare = np.sum(value_counts <= len(df[col]) * 0.01)
    cib_ratio = value_counts.max()  / value_counts.min()
    top_coverage = df[col].value_counts(normalize=True).head(5).sum()

    if n > 0:
        entropy = -(frequencies * np.log2(frequencies)).sum()
        gini = 1 - (frequencies ** 2).sum()
        simpson = 1 / (frequencies ** 2).sum()
        cardinality_ratio = df[col].nunique() / n
        mode_value = df[col].mode().iloc[0]
    else:
        entropy = np.nan
        gini = np.nan
        simpson = np.nan
        cardinality_ratio = np.nan
        mode_value = ""
    lengths = df[col].astype(str).str.len()

    return CategoricalColumns(
        name=col,
        unique_categories=df[col].nunique(),
        mode=mode_value,
        entropy=round(entropy, 2),
        frequencies=df[col].value_counts(normalize=True).head(20).to_dict(),
        gini=round(gini, 2),
        simpson_diversity=round(simpson, 2),
        value_counts=df[col].value_counts().head(20).to_dict(),
        max_category_length=lengths.max(),
        min_category_length=lengths.min(),
        cardinality_ratio=round(cardinality_ratio, 3),
        rare_categories=rare,
        top_5_coverage=top_coverage,
        cib_ratio=cib_ratio,
        top_1_coverage=(df[col].value_counts(normalize=True).max() if n > 0 else np.nan), # Near Zero Variance (NZV)
        effective_cardinality=((df[col].nunique()/len(df[col]))*100),
        memory = df[col].memory_usage(deep=True),
    )
