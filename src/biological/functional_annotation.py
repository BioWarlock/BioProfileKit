from dataclasses import dataclass
import pandas as pd
from data_utils.remote_data import get_clusters_of_orthologous_groups, get_gene_ontology

@dataclass
class AnnotationFlags:
    name: str
    is_annotation: bool
    valid_annotation: set | str | None


def annotation_flags(df, col, annotation_type) -> AnnotationFlags | None:
    if annotation_type == "cog":
        cog_df = get_clusters_of_orthologous_groups()
        results = validate_annotation(df[col], cog_df, "COG_ID")
    elif annotation_type == "go":
        go_df = get_gene_ontology()
        results = validate_annotation(df[col], go_df, "GO_ID")
    else:
        raise ValueError(f"Unknown annotation type: {annotation_type}")

    return AnnotationFlags(
        name=col,
        is_annotation=results is not None,
        valid_annotation=results
    )


def validate_annotation(col: pd.Series, annotation_df: pd.DataFrame, id_column: str,
                        threshold: float = 0.8) -> set | str | None:
    col_cleaned = clean_strings(col)
    valid_annotations = set(clean_strings(annotation_df[id_column]))

    is_valid = col_cleaned.isin(valid_annotations)
    validity_rate = is_valid.mean()

    if validity_rate < threshold:
        is_valid_raw = col.isin(annotation_df[id_column])
        validity_rate_raw = is_valid_raw.mean()
        if validity_rate_raw > validity_rate:
            is_valid = is_valid_raw
            validity_rate = validity_rate_raw

    if validity_rate > threshold:
        invalid_annotations = pd.Series(col.values)[~is_valid].unique()
        if invalid_annotations.size > 0:
            return set(invalid_annotations.astype(str))
        else:
            return "Valid"

    return None

def clean_strings(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip().str.upper()