from dataclasses import dataclass
from typing import Tuple

import pandas as pd

@dataclass
class TaxonomyFlags:
    name: str
    is_taxonomy: bool
    taxid: set | str | None
    taxonomy: list | str | None
    rank_distribution: dict | None = None
    is_mixed: bool = False
    invalid_names: list | None = None
    outdated_names : dict | None = None


def taxonomy_flags(df, col, valid_names, valid_tax_ids, name_to_rank, taxid_to_rank,
                   name_to_scientific) -> TaxonomyFlags:
    if df[col].dtype in ['int64', 'float64'] or pd.api.types.is_numeric_dtype(df[col]):
        taxid_result = is_taxid(df[col], valid_tax_ids)
        if taxid_result is not None:
            distribution, is_mixed = taxid_rank_distribution(df[col], taxid_to_rank)
            return TaxonomyFlags(
                name=col,
                is_taxonomy=True,
                taxid=taxid_result,
                taxonomy=None,
                rank_distribution=distribution,
                is_mixed=is_mixed,
                invalid_names=None,
            )
    else:
        taxonomy_result = is_taxonomy(df[col], valid_names, name_to_rank, name_to_scientific)
        if taxonomy_result is not None:
            return TaxonomyFlags(
                name=col,
                is_taxonomy=True,
                taxid=None,
                taxonomy=taxonomy_result["invalid_names"] or "Valid",
                rank_distribution=taxonomy_result["rank_distribution"],
                is_mixed=taxonomy_result["is_mixed"],
                invalid_names=taxonomy_result["invalid_names"],
                outdated_names=taxonomy_result["outdated"],
            )
    return TaxonomyFlags(
        name=col,
        is_taxonomy=False,
        taxid=None,
        taxonomy=None,
        rank_distribution=None,
        is_mixed=False,
        invalid_names=None
    )


def build_lookups(vocab: pd.DataFrame) -> Tuple[set, set, dict, dict, dict]:
    valid_names = set(vocab['name_txt'])
    valid_tax_ids = set(vocab['tax_id'])
    name_to_rank = dict(zip(vocab['name_txt'], vocab['rank']))
    name_to_scientific = dict(zip(vocab['name_txt'], vocab['scientific_name']))

    sci = vocab[vocab['name_class'] == 'scientific name'].drop_duplicates('tax_id')
    taxid_to_rank = dict(zip(sci['tax_id'], sci['rank']))

    return valid_names, valid_tax_ids, name_to_rank, taxid_to_rank, name_to_scientific


def is_taxid(col: pd.Series, valid_tax_ids: set, threshold: float = 0.9) -> set | str | None:
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


def is_taxonomy(col: pd.Series, valid_names: set, name_to_rank: dict, name_to_scientific: dict, threshold: float = 0.8) -> dict | None:
    col_obj = col.astype(object)
    uniques = pd.unique(col_obj.dropna())
    valid_unique = {u for u in uniques if u in valid_names}

    is_valid = col_obj.isin(valid_unique)
    validity_rate = is_valid.sum() / len(col)

    cleaned_names = col_obj
    if validity_rate < threshold:
        cleaned_names = col_obj.astype(str).str.extract(r'^([^(]+)')[0].str.strip()
        cleaned_uniques = pd.unique(cleaned_names.dropna())
        valid_cleaned_unique = {u for u in cleaned_uniques if u in valid_names}
        is_valid_cleaned = cleaned_names.isin(valid_cleaned_unique)
        validity_rate_cleaned = is_valid_cleaned.sum() / len(col)

        if validity_rate_cleaned > validity_rate:
            is_valid = is_valid_cleaned
            validity_rate = validity_rate_cleaned

    if validity_rate > threshold:
        distribution, is_mixed, invalid_names = rank_distribution(cleaned_names, name_to_rank)
        outdated = find_outdated_names(cleaned_names, valid_names, name_to_scientific)
        return{
            "valid":True,
            "rank_distribution":distribution,
            "is_mixed":is_mixed,
            "invalid_names":invalid_names if invalid_names else None,
            "outdated":outdated if outdated else None,
        }
    return None


def taxid_rank_distribution(col: pd.Series, taxid_to_rank: dict, threshold: float = 0.05) -> Tuple[dict, bool]:
    numeric = pd.to_numeric(col, errors='coerce')
    ranks = numeric.map(taxid_to_rank)

    rank_counts = ranks.value_counts(normalize=True)
    distribution = {rank: round(float(freq), 4) for rank, freq in rank_counts.items()}

    is_mixed = len([f for f in distribution.values() if f >= threshold]) > 1
    return distribution, is_mixed

def rank_distribution(col: pd.Series, name_to_rank: dict, threshold: float = 0.05) -> Tuple[dict, bool, list]:
    cleaned = col.astype(str).str.strip()
    ranks = cleaned.map(name_to_rank)

    invalid_names = cleaned[ranks.isna() & col.notnull()].unique().tolist()

    rank_counts = ranks.value_counts(normalize=True)
    distribution = {rank: round(float(freq), 4) for rank, freq in rank_counts.items()}

    is_mixed = len([f for f in distribution.values() if f >= threshold]) > 1
    return distribution, is_mixed, sorted(invalid_names)


def find_outdated_names(col: pd.Series, valid_names: set, name_to_scientific: dict) -> dict:
    cleaned = col.astype(str).str.strip()
    uniques = pd.unique(cleaned.dropna())
    valid_used = [u for u in uniques if u in valid_names]

    outdated = {}
    for name in valid_used:
        current = name_to_scientific.get(name)
        if current is not None and name != current:
            outdated[name] = current
    return outdated