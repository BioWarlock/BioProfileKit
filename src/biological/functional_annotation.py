from dataclasses import dataclass

import pandas as pd
import plotly.graph_objs as go

from data_utils.remote_data import get_clusters_of_orthologous_groups, get_gene_ontology
from enums import COGColor, COG_GROUP_ORDER

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
class AnnotationFlags:
    name: str
    is_annotation: bool
    valid_annotation: set | str | None
    cog_bar: str | None
    cog_group: str | None
    cog_invalid: list | None


def build_annotation_lookup(annotation_df: pd.DataFrame, id_column: str) -> dict:
    return {
        "cleaned": set(clean_strings(annotation_df[id_column])),
        "raw": set(annotation_df[id_column].astype(str)),
    }


def annotation_flags(df, col, annotation_type) -> AnnotationFlags | None:
    cog_barchart = None
    cog_group = None
    cog_invalid = None
    if annotation_type == "cog":
        cog_df = get_clusters_of_orthologous_groups()
        id_results = validate_annotation(df[col], cog_df, "COG_ID")
        name_results = validate_annotation(df[col], cog_df, "COG name")
        if id_results is not None:
            results = id_results
            matched_column = "COG_ID"
        elif name_results is not None:
            results = name_results
            matched_column = "COG name"
        else:
            results = None
            matched_column = None

        if matched_column is not None:
            counts_df, cog_invalid = build_cog_counts(df[col], cog_df, cog_col=matched_column)
            cog_barchart = cog_category_barchart(counts_df)
            cog_group = cog_group_donut(counts_df)

    elif annotation_type == "go":
        go_df = get_gene_ontology()
        results = validate_annotation(df[col], go_df, "GO_ID")
    else:
        raise ValueError(f"Unknown annotation type: {annotation_type}")

    return AnnotationFlags(
        name=col,
        is_annotation=results is not None,
        valid_annotation=results,
        cog_bar=cog_barchart,
        cog_group=cog_group,
        cog_invalid=cog_invalid
    )


def validate_annotation(col: pd.Series, annotation_df: pd.DataFrame, category: str, threshold: float = 0.8) -> set | str | None:
    cleaned = clean_strings(col)
    valid_annotations = set(clean_strings(annotation_df[category]))

    is_valid = cleaned.isin(valid_annotations)
    validity_rate = is_valid.mean()

    if validity_rate < threshold:
        is_valid_raw = col.isin(annotation_df[category])
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


def build_cog_counts(col: pd.Series, cog__df: pd.DataFrame, cog_col: str = "COG_ID") -> tuple[pd.DataFrame, list | None]:
    cog_categories = cog__df[[cog_col, "Functional Category"]].copy()
    cog_categories["category"] = cog_categories["Functional Category"].str.strip().apply(list)
    cog_categories = cog_categories.explode("category")[[cog_col, "category"]]

    values = col.dropna().astype(str).str.strip().to_frame(cog_col)
    invalid_values = sorted(set(values[cog_col].unique()) - set(cog_categories[cog_col]))
    print(invalid_values)
    invalid_values = invalid_values if invalid_values else None
    matched = values.merge(cog_categories, on=cog_col, how="inner")

    if matched.empty:
        return pd.DataFrame(columns=["category", "group", "color", "description", "count"]), None

    counts = matched["category"].value_counts().rename_axis("category").reset_index(name="count")

    meta = pd.DataFrame([(c.name, c.group, c.color, c.description) for c in COGColor], columns=["category", "group", "color", "description"])
    df= counts.merge(meta, on="category", how="left")[["category", "group", "color", "description", "count"]]
    return df, invalid_values


def cog_category_barchart(counts: pd.DataFrame):
    if counts.empty:
        return None

    ordered = counts.sort_values(by="group", key=lambda s: s.map({g: i for i, g in enumerate(COG_GROUP_ORDER)})).sort_values("category")

    fig = go.Figure(go.Bar(
        x=ordered["category"],
        y=ordered["count"],
        marker_color=ordered["color"],
        marker_line=dict(color="#888", width=1),
        text=ordered["count"],
        textposition="outside",
        hovertext=ordered["description"],
        hovertemplate="<b>%{x}</b>: %{hovertext}<br>Count: %{y}<extra></extra>"
    ))
    fig.update_layout(
        title="COG Functional Category Distribution",
        xaxis_title="Category",
        yaxis_title="Count",
        template="plotly_white"
    )
    config['toImageButtonOptions']['filename'] = "cog_category_barchart"
    return fig.to_html(full_html=False, include_plotlyjs=False, config=config)


def cog_group_donut(counts: pd.DataFrame):
    if counts.empty:
        return None

    group_counts = counts.groupby("group")["count"].sum().reindex(COG_GROUP_ORDER).dropna().reset_index()

    group_colors = {
        "INFORMATION STORAGE AND PROCESSING": "#F9A8D4",
        "CELLULAR PROCESSES AND SIGNALING": "#BEF264",
        "METABOLISM": "#93C5FD",
        "POORLY CHARACTERIZED": "#D1D5DB",
    }

    fig = go.Figure(go.Pie(
        labels=group_counts["group"],
        values=group_counts["count"],
        hole=0.5,
        marker=dict(colors=[group_colors[g] for g in group_counts["group"]]),
        textinfo="percent",
        hovertemplate="<b>%{label}</b><br>Count: %{value}<br>Share: %{percent}<extra></extra>",
    ))
    fig.update_layout(
        title="COG Group Distribution",
        template="plotly_white",
        legend=dict(orientation="h", y=-0.15),
    )
    config['toImageButtonOptions']['filename'] = "cog_group_distribution"
    return fig.to_html(full_html=False, include_plotlyjs=False, config=config)
