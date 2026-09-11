import pandas as pd
from dataclasses import dataclass


@dataclass
class UniProtFlags:
    name: str
    is_uniprot: bool
    match_type: str | None
    validity_rate: float | None = None
    invalid_values: list | None = None
    outdated_names: dict | None = None


def uniprot_flags(df, col, lookups: dict, threshold: float = 0.8, min_unique: int = 2) -> UniProtFlags:
    series = df[col]

    if pd.api.types.is_numeric_dtype(series):
        return UniProtFlags(name=col, is_uniprot=False, match_type=None)

    non_null = series.dropna()
    if len(non_null) == 0 or non_null.nunique() < min_unique:
        return UniProtFlags(name=col, is_uniprot=False, match_type=None)

    accession_result = is_uniprot_accession(non_null, lookups, threshold)
    if accession_result is not None:
        return UniProtFlags(
            name=col, is_uniprot=True, match_type="accession",
            validity_rate=accession_result["validity_rate"],
            invalid_values=accession_result["invalid_values"],
        )

    protein_result = is_uniprot_protein_name(non_null, lookups, threshold)
    if protein_result is not None:
        return UniProtFlags(
            name=col, is_uniprot=True, match_type="protein_name",
            validity_rate=protein_result["validity_rate"],
            invalid_values=protein_result["invalid_names"],
            outdated_names=protein_result["outdated_names"],
        )

    non_null_str = non_null.astype(str).str.strip()
    n_non_null = len(non_null_str)

    is_valid_ec = non_null_str.isin(lookups['ec_numbers'])
    validity_ec = is_valid_ec.sum() / n_non_null
    if validity_ec > threshold:
        invalid = non_null_str[~is_valid_ec].unique().tolist()
        return UniProtFlags(
            name=col, is_uniprot=True, match_type="ec_number",
            validity_rate=round(float(validity_ec), 4),
            invalid_values=sorted(invalid) if invalid else None,
        )

    is_valid_gene = non_null_str.isin(lookups['gene_names'])
    validity_gene = is_valid_gene.sum() / n_non_null
    if validity_gene > threshold:
        invalid = non_null_str[~is_valid_gene].unique().tolist()
        return UniProtFlags(
            name=col, is_uniprot=True, match_type="gene_name",
            validity_rate=round(float(validity_gene), 4),
            invalid_values=sorted(invalid) if invalid else None,
        )

    return UniProtFlags(name=col, is_uniprot=False, match_type=None)


def is_uniprot_accession(col: pd.Series, lookups: dict, threshold: float) -> dict | None:
    non_null = col.astype(str).str.strip()
    n_non_null = len(non_null)

    is_valid = non_null.isin(lookups['accessions'])
    validity_rate = is_valid.sum() / n_non_null

    if validity_rate <= threshold:
        return None

    invalid = non_null[~is_valid].unique().tolist()
    return {
        "validity_rate": round(float(validity_rate), 4),
        "invalid_values": sorted(invalid) if invalid else None,
    }


def is_uniprot_protein_name(col: pd.Series, lookups: dict, threshold: float) -> dict | None:
    non_null = col.astype(str).str.strip()
    n_non_null = len(non_null)

    is_valid = non_null.isin(lookups['protein_names'])
    validity_rate = is_valid.sum() / n_non_null
    matched = non_null

    if validity_rate <= threshold:
        cleaned = non_null.str.extract(r'^([^(]+)')[0].str.strip()
        cleaned_lower = cleaned.str.lower()
        is_valid_cleaned = cleaned_lower.isin(lookups['protein_names_lower'])
        validity_rate_cleaned = is_valid_cleaned.sum() / n_non_null

        if validity_rate_cleaned > validity_rate:
            is_valid = is_valid_cleaned
            validity_rate = validity_rate_cleaned
            matched = cleaned

    if validity_rate <= threshold:
        return None

    invalid_names = non_null[~is_valid].unique().tolist()
    valid_values = matched[is_valid].unique()
    outdated = {
        v: lookups['name_to_primary'][v.lower()]
        for v in valid_values
        if lookups['name_to_primary'].get(v.lower(), v) != v
    }

    return {
        "validity_rate": round(float(validity_rate), 4),
        "invalid_names": sorted(invalid_names) if invalid_names else None,
        "outdated_names": outdated if outdated else None,
    }

def build_uniprot_lookups(df: pd.DataFrame) -> dict:
    accessions = set(df['accession'].dropna())

    base = df[['accession', 'protein_name']].dropna(subset=['protein_name'])
    name_to_primary = dict(zip(base['protein_name'], base['protein_name']))

    for col in ('alt_names', 'short_names'):
        variants = df[['protein_name', col]].dropna(subset=[col]).copy()
        variants[col] = variants[col].str.split(';')
        variants = variants.explode(col)
        variants[col] = variants[col].str.strip()

        for variant, primary in zip(variants[col], variants['protein_name']):
            name_to_primary.setdefault(variant, primary)

    protein_names = set(name_to_primary.keys())
    protein_names_lower = {n.lower() for n in protein_names}
    name_to_primary_lower = {k.lower(): v for k, v in name_to_primary.items()}

    ec_numbers = set(df['ec_numbers'].dropna().str.split(';').explode().str.strip())
    gene_names = set(df['gene_names'].dropna().str.split(';').explode().str.strip())

    return {
        "accessions": accessions,
        "protein_names": protein_names,
        "protein_names_lower": protein_names_lower,
        "name_to_primary": name_to_primary_lower,
        "ec_numbers": ec_numbers,
        "gene_names": gene_names,
    }

#ToDo: remove after debuggin/testing
def diagnose_protein_name_mismatch(col: pd.Series, lookups: dict) -> dict:
    non_null = col.dropna().astype(str).str.strip()
    uniques = pd.Series(non_null.unique())

    exact_match = uniques.isin(lookups['protein_names'])
    lower_match = uniques.str.lower().isin(lookups['protein_names_lower']) & ~exact_match

    still_invalid = uniques[~exact_match & ~lower_match]

    cleaned = still_invalid.str.extract(r'^([^(]+)')[0].str.strip().str.lower()
    cleaned_match = cleaned.isin(lookups['protein_names_lower'])

    return {
        "n_unique_total": len(uniques),
        "n_exact_match": int(exact_match.sum()),
        "n_lower_match_only": int(lower_match.sum()),
        "n_still_invalid": len(still_invalid),
        "n_would_match_after_cleaning": int(cleaned_match.sum()),
        "examples_still_invalid": still_invalid[~cleaned_match.values].head(20).tolist(),
    }