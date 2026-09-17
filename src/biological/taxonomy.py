from dataclasses import dataclass
from typing import Tuple

import pandas as pd
import plotly.express as px

SUNBURST_RANKS = ["domain", "kingdom", "phylum", "class", "order", "family", "genus", "species"]
SUNBURST_PATH = ["superdomain"] + SUNBURST_RANKS
RANK_ALIASES = {"superkingdom": "domain", "realm": "domain"}

config = {
    'toImageButtonOptions': {
        'format': 'png',
        'filename': None,
        'height': 1200,
        'width': 1200,
        'scale': 4
    }
}


@dataclass
class TaxonomyFlags:
    name: str
    is_taxonomy: bool
    taxid: set | str | None
    taxonomy: list | str | None
    rank_distribution: dict | None = None
    is_mixed: bool = False
    invalid_names: list | None = None
    outdated_names: dict | None = None
    sunburst_plot: str | None = None


def taxonomy_flags(df, col, valid_names, valid_tax_ids, name_to_rank, taxid_to_rank, name_to_scientific, name_to_taxid=None, lineage_resolver=None) -> TaxonomyFlags:
    if df[col].dtype in ['int64', 'float64'] or pd.api.types.is_numeric_dtype(df[col]):
        taxid_result = is_taxid(df[col], valid_tax_ids)
        if taxid_result is not None:
            distribution, is_mixed = taxid_rank_distribution(df[col], taxid_to_rank)
            sunburst_plot = None
            if lineage_resolver is not None:
                needed_taxids = set(pd.to_numeric(df[col], errors='coerce').dropna().astype(int).unique())
                lineage_table = lineage_resolver.build_lineage(needed_taxids)
                sunburst_data = build_sunburst_data_from_taxids(df[col], lineage_table)
                if not sunburst_data.empty:
                    sunburst_plot = taxonomy_sunburst_plot(sunburst_data)
            return TaxonomyFlags(
                name=col,
                is_taxonomy=True,
                taxid=taxid_result,
                taxonomy=None,
                rank_distribution=distribution,
                is_mixed=is_mixed,
                invalid_names=None,
                sunburst_plot=sunburst_plot
            )
    else:
        taxonomy_result = is_taxonomy(df[col], valid_names, name_to_rank, name_to_scientific)
        if taxonomy_result is not None:
            sunburst_plot = None
            if name_to_taxid is not None and lineage_resolver is not None:
                needed_taxids = {
                    name_to_taxid[n] for n in df[col].dropna().astype(str).str.strip().unique()
                    if n in name_to_taxid
                }
                lineage_table = lineage_resolver.build_lineage(needed_taxids)
                sunburst_data = build_sunburst_data(df[col], name_to_taxid, lineage_table)
                if not sunburst_data.empty:
                    sunburst_plot = taxonomy_sunburst_plot(sunburst_data)
            return TaxonomyFlags(
                name=col,
                is_taxonomy=True,
                taxid=None,
                taxonomy=taxonomy_result["invalid_names"] or "Valid",
                rank_distribution=taxonomy_result["rank_distribution"],
                is_mixed=taxonomy_result["is_mixed"],
                invalid_names=taxonomy_result["invalid_names"],
                outdated_names=taxonomy_result["outdated"],
                sunburst_plot=sunburst_plot
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
    tmp = vocab.drop_duplicates('name_txt', keep='last').set_index('name_txt')
    name_to_rank = tmp['rank']
    name_to_scientific = tmp['scientific_name']
    sci = vocab[vocab['name_class'] == 'scientific name'].drop_duplicates('tax_id')
    taxid_to_rank = sci.set_index('tax_id')['rank']

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
    tmp = col.astype(object)
    uniques = pd.unique(tmp.dropna())
    valid_unique = {u for u in uniques if u in valid_names}

    is_valid = tmp.isin(valid_unique)
    validity_rate = is_valid.sum() / len(col)
    names = pd.Series([])
    if validity_rate < threshold:
        names = tmp.astype(str).str.extract(r'^([^(]+)')[0].str.strip()
        cleaned_uniques = pd.unique(names.dropna())
        valid_cleaned_unique = {u for u in cleaned_uniques if u in valid_names}
        is_valid = names.isin(valid_cleaned_unique)
        validity_rate_cleaned = is_valid.sum() / len(col)

        if validity_rate_cleaned > validity_rate:
            validity_rate = validity_rate_cleaned

    if validity_rate > threshold:
        distribution, is_mixed, invalid_names = rank_distribution(names, name_to_rank)
        outdated = find_outdated_names(names, valid_names, name_to_scientific)
        return {
            "valid": True,
            "rank_distribution": distribution,
            "is_mixed": is_mixed,
            "invalid_names": invalid_names if invalid_names else None,
            "outdated": outdated if outdated else None,
        }
    return None


def taxid_rank_distribution(col: pd.Series, taxid_to_rank: dict, threshold: float = 0.05) -> Tuple[dict, bool]:
    rank_counts = pd.to_numeric(col, errors='coerce').map(taxid_to_rank).value_counts(normalize=True)
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
    valids = [u for u in uniques if u in valid_names]

    outdated = {}
    for name in valids:
        current = name_to_scientific.get(name)
        if current is not None and name != current:
            outdated[name] = current
    return outdated


def build_name_to_taxid(df: pd.DataFrame) -> dict:
    return df.drop_duplicates('name_txt', keep='last').set_index('name_txt')['tax_id']


def build_sunburst_data(col: pd.Series, name_to_taxid: dict, lineage_table: pd.DataFrame) -> pd.DataFrame:
    counts = col.dropna().astype(str).str.strip().value_counts().reset_index()
    counts.columns = ["name", "count"]
    counts["tax_id"] = counts["name"].map(name_to_taxid)
    counts = counts.dropna(subset=["tax_id"])

    if counts.empty or lineage_table.empty:
        return pd.DataFrame(columns=SUNBURST_PATH + ["count"])

    df = counts.merge(lineage_table, on="tax_id", how="left")

    return df[SUNBURST_PATH + ["count"]].dropna(subset=SUNBURST_PATH, how="all")


def build_sunburst_data_from_taxids(col: pd.Series, lineage_table: pd.DataFrame) -> pd.DataFrame:
    counts = pd.to_numeric(col, errors='coerce').dropna().astype(int).value_counts().reset_index()
    counts.columns = ["tax_id", "count"]

    if counts.empty or lineage_table.empty:
        return pd.DataFrame(columns=SUNBURST_PATH + ["count"])

    df = counts.merge(lineage_table, on="tax_id", how="left")

    return df[SUNBURST_PATH + ["count"]].dropna(subset=SUNBURST_PATH, how="all")


def taxonomy_sunburst_plot(sunburst_df: pd.DataFrame):
    fig = px.sunburst(
        sunburst_df,
        path=SUNBURST_PATH,
        values="count",
        color="superdomain",
        color_discrete_map={
            "Viruses": "#0F65A0",
            "Bacteria": "#994564",
            "Eukaryota": "#65A1E1",
        },
        maxdepth=4,
        height=700,
    )
    fig.update_traces(
        hovertemplate="<b>%{label}</b><br>Count: %{value}<br>Share of parent: %{percentParent:.1%}<br>Share of total: %{percentRoot:.1%}<extra></extra>",
        textfont=dict(color="white"),
        insidetextfont=dict(color="white")
    )
    fig.update_layout(title="Taxonomic Composition")
    config['toImageButtonOptions']['filename'] = "taxonomic_composition"
    return fig.to_html(full_html=False, include_plotlyjs=False, config=config)
