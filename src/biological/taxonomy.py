from dataclasses import dataclass
import pandas as pd

@dataclass
class TaxonomyFlags:
    name: str
    is_taxonomy: bool
    taxid: set | str | None
    taxonomy: list | str | None


def taxonomy_flags(df, col, tax_df) -> TaxonomyFlags:

    if df[col].dtype in ['int64', 'float64'] or pd.api.types.is_numeric_dtype(df[col]):
        taxid_result = is_taxid(df[col], tax_df)
        if taxid_result is not None:
            return TaxonomyFlags(
                name=col,
                is_taxonomy=True,
                taxid=taxid_result,
                taxonomy=None
            )
    else:
        taxonomy_result = is_taxonomy(df[col], tax_df)
        if taxonomy_result is not None:
            return TaxonomyFlags(
                name=col,
                is_taxonomy=True,
                taxid=None,
                taxonomy=taxonomy_result
            )

    return TaxonomyFlags(
        name=col,
        is_taxonomy=False,
        taxid=None,
        taxonomy=None
    )


def is_taxid(col: pd.Series, tax_df: pd.DataFrame, threshold: float = 0.9) -> set | str | None:
    valid_tax_ids = set(tax_df['tax_id'])
    excluded_cols = ["length", "start", "end"]

    if col.name and str(col.name).lower() in excluded_cols:
        return None

    tmp_series = pd.to_numeric(col, errors='coerce')
    is_numeric_candidate = tmp_series.notna().sum() / len(col) > threshold

    if is_numeric_candidate:
        is_valid = tmp_series.isin(valid_tax_ids)
        validity_rate = is_valid.sum() / len(col)

        if validity_rate > threshold:
            invalid_mask = ~is_valid & tmp_series.notna()
            invalid_ids = set(col.loc[invalid_mask].tolist())

            if invalid_ids:
                return invalid_ids
            else:
                return "all tax IDs valid"

    return None


def is_taxonomy(col: pd.Series, tax_df: pd.DataFrame, threshold: float = 0.8) -> list | str | None:
    valid_names = set(tax_df['name_txt'])
    is_valid = col.isin(valid_names)
    validity_rate = is_valid.sum() / len(col)
    if validity_rate < threshold:
        cleaned_names = col.astype(str).str.extract(r'^([^(]+)')[0].str.strip()
        is_valid_cleaned = cleaned_names.isin(valid_names)
        validity_rate_cleaned = is_valid_cleaned.sum() / len(col)

        if validity_rate_cleaned > validity_rate:
            is_valid = is_valid_cleaned
            validity_rate = validity_rate_cleaned

    if validity_rate > threshold:
        invalid_names_list = col.loc[~is_valid & col.notnull()].unique().tolist()
        if invalid_names_list:
            return sorted(invalid_names_list)
        else:
            return "Valid"
    return None
