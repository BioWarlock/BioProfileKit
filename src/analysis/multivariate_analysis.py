import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from scipy.stats import chi2_contingency

from models.multivariate import MultivariateAnalysis
from .plot_utils import apply_standard_axes


def multivariate_analysis(df: pd.DataFrame, target: str) -> MultivariateAnalysis:
    values, methods = compute_correlation_matrix(df)
    feat_target_corr = feature_target_correlation(df, target) if target else None

    return MultivariateAnalysis(
        correlation_heatmap=correlation_heatmap(values, methods),
        pearson_heatmap=pearson_correlation_heatmap(df),
        cramers_heatmap=cramers_heatmap(df),
        eta_heatmap=eta_heatmap(df),
        missing_matrix=missing_matrix(df),
        missing_values_barchart=missing_values_barchart(df),
        balance_plot=balance_plot(df, target) if target else None,
        boxplot=boxplot(df),
        scatter_matrix=scatter_matrix(df),
        correlation_matrix=values,
        correlation_methods=methods,
        top_associations=top_associations(values, methods), #ToDo: Print Tabelle
        feature_target_correlation=feat_target_corr, # ToDo: Print Tabelle neben Plot
        feature_target_plot=feature_target_plot(feat_target_corr) if feat_target_corr else None, # ToDo: der Plot
        mutual_information=None,
        mcar_result=littles_mcar_test(df),
    )

def _classify_columns(df: pd.DataFrame) -> dict:
    """Return {column: 'numeric' | 'categorical'} for usable columns.

    Drops constant numeric columns and single-value categoricals, since they
    carry no association signal.
    """
    num = df.select_dtypes(include='number')
    num_std = num.std(ddof=0)
    numeric = num_std[num_std > 0].index

    cat = df.select_dtypes(exclude='number')
    cat_nunique = cat.nunique(dropna=True)
    categorical = cat_nunique[cat_nunique > 1].index

    return {**{c: 'numeric' for c in numeric}, **{c: 'categorical' for c in categorical}}
"""
Correlation
"""
# Num x Num
def _pearson(df: pd.DataFrame, a: str, b: str) -> float:
    pair = df[[a, b]].dropna()
    if len(pair) < 2:
        return np.nan
    return pair[a].corr(pair[b], method='pearson')

# Cat x Cat
def _cramers_v(df: pd.DataFrame, a: str, b: str) -> float:
    """Bias-corrected Cramér's V (Bergsma 2013)."""
    pair = df[[a, b]].dropna()
    if pair.empty:
        return np.nan

    confusion = pd.crosstab(pair[a], pair[b])
    if confusion.shape[0] < 2 or confusion.shape[1] < 2:
        return np.nan

    chi2 = chi2_contingency(confusion)[0]
    n = confusion.to_numpy().sum()
    phi2 = chi2 / n
    r, k = confusion.shape

    # Bergsma-Bias Correction
    phi2corr = max(0.0, phi2 - ((k - 1) * (r - 1)) / (n - 1))
    rcorr = r - ((r - 1) ** 2) / (n - 1)
    kcorr = k - ((k - 1) ** 2) / (n - 1)

    denom = min((kcorr - 1), (rcorr - 1))
    if denom <= 0:
        return np.nan
    return np.sqrt(phi2corr / denom)

# Num x Cat
def _eta_squared(df: pd.DataFrame, cat: str, num: str) -> float:
    """Correlation ratio eta^2 (ANOVA): variance in num explained by groups of cat."""
    pair = df[[cat, num]].dropna()
    if len(pair) < 2:
        return np.nan

    groups = pair.groupby(cat)[num]
    grand_mean = pair[num].mean()

    ss_between = sum(len(g) * (g.mean() - grand_mean) ** 2 for _, g in groups)
    ss_total = ((pair[num] - grand_mean) ** 2).sum()

    if ss_total == 0:
        return np.nan
    return ss_between / ss_total


def _correlation_pair(df, a, b, types) -> tuple[float, str]:
    ta, tb = types[a], types[b]

    if ta == 'numeric' and tb == 'numeric':
        return abs(_pearson(df, a, b)), 'Pearson'
    if ta == 'categorical' and tb == 'categorical':
        return _cramers_v(df, a, b), "Cramér's V"
    cat, num = (a, b) if ta == 'categorical' else (b, a)
    return _eta_squared(df, cat, num), 'Eta²'


def compute_correlation_matrix(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    types = _classify_columns(df)
    cols = list(types.keys())
    n = len(cols)

    values = pd.DataFrame(np.eye(n), index=cols, columns=cols)
    methods = pd.DataFrame('', index=cols, columns=cols, dtype=object)
    for c in cols:
        methods.at[c, c] = 'self'

    for i in range(n):
        for j in range(i + 1, n):
            a, b = cols[i], cols[j]
            value, method = _correlation_pair(df, a, b, types)
            value = round(float(value), 3) if value == value else np.nan

            values.iat[i, j] = value
            values.iat[j, i] = value
            methods.iat[i, j] = method
            methods.iat[j, i] = method

    return values, methods


def feature_target_correlation(df: pd.DataFrame, target: str) -> dict | None:
    if target not in df.columns:
        return None

    types = _classify_columns(df)
    if target not in types:
        return None

    result = {}
    for col in types:
        if col == target:
            continue
        value, method = _correlation_pair(df, col, target, types)
        if value == value:
            result[col] = {'value': round(float(value), 3), 'method': method}
    return result or None

def feature_target_plot(feature_target: dict | None):
    if not feature_target:
        return None

    ranked = sorted(feature_target.items(), key=lambda x: x[1]['value'])
    features = [f for f, _ in ranked]
    valuevec = [info['value'] for _, info in ranked]
    methodvec = [info['method'] for _, info in ranked]

    method_colors = {
        "Pearson": "#0F65A0",
        "Cramér's V": "#65A1E1",
        "Eta squared": "#994564",
    }
    bar_colors = [method_colors.get(m, "#A1ACBD") for m in methodvec]

    fig = go.Figure()
    seen = set()
    for f, v, m, c in zip(features, valuevec, methodvec, bar_colors):
        fig.add_trace(go.Bar(
            x=[v], y=[f], orientation='h',
            marker_color=c, name=m,
            legendgroup=m, showlegend=m not in seen,
            text=[f"{v:.3f}"], textposition="outside",
            hovertemplate=f"{f}<br>Association: {v:.3f}<br>Method: {m}<extra></extra>",
        ))
        seen.add(m)

    fig.update_layout(
        title="Feature–Target Association",
        xaxis=dict(title="Association strength", range=[0, 1.05]),
        yaxis=dict(title="Feature"),
        template="plotly_white",
        bargap=0.3,
        height=max(400, 40 * len(features) + 150),
        legend=dict(title="Method"),
    )
    fig.show()
    return fig.to_html(full_html=False, include_plotlyjs=False)

#ToDo change for Column Overview
def get_correlation(df: pd.DataFrame, col) -> list | None:
    ncols = df.select_dtypes(include='number').dropna(axis=1, how='all').columns
    if col in ncols:
        std = df[ncols].std(ddof=0)
        ncols = std[std > 0].index
        if col not in ncols:
            return None
        corr = df[ncols].corrwith(df[col], method='pearson')
        corr.drop(labels=col, inplace=True)
        corr = corr[corr.abs() >= 0.3]
        if corr.empty:
            return None
        return list(zip(corr.index, corr))
    return None

def pearson_correlation_heatmap(df: pd.DataFrame):
    df_numeric = df.select_dtypes(include=['float64', 'int64']).dropna(axis=1, how='all')
    if not df_numeric.empty:
        std = df_numeric.std(ddof=0)
        df_numeric = df_numeric.loc[:, std[std > 0].index]
    corr_matrix = round(df_numeric.corr(), 3)
    fig = px.imshow(corr_matrix, text_auto=True,
                    labels=dict(color="Correlation"),
                    color_continuous_scale="RdBu_r", aspect="auto", height=700)
    fig.update_layout(title="Correlation Heatmap")
    return fig.to_html(full_html=False, include_plotlyjs=False)

def correlation_heatmap(df: pd.DataFrame, methods: pd.DataFrame):
    fig = px.imshow(
        df, text_auto=True,
        labels=dict(color="Association"),
        color_continuous_scale="Blues", aspect="auto", height=700,
        zmin=0, zmax=1,
    )
    fig.update_traces(
        customdata=methods.values,
        hovertemplate="%{x} ↔ %{y}<br>Association: %{z}<br>Method: %{customdata}<extra></extra>",
    )
    fig.update_layout(
        title="Association Heatmap (mixed-type)",
        margin=dict(b=120),
    )
    return fig.to_html(full_html=False, include_plotlyjs=False)


def cramers_heatmap(df: pd.DataFrame) -> str | None:
    """Cramér's V matrix over categorical columns only (0..1)."""
    cols = [c for c, t in _classify_columns(df).items() if t == 'categorical']
    if len(cols) < 2:
        return None

    mat = pd.DataFrame(np.eye(len(cols)), index=cols, columns=cols)
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            val = _cramers_v(df, cols[i], cols[j])
            val = round(float(val), 3) if val == val else np.nan
            mat.iat[i, j] = val
            mat.iat[j, i] = val

    # light -> primary blue (#0F65A0)
    colorscale = [
        [0.0, "#EAF4FB"], [0.25, "#A9E1FF"], [0.5, "#65A1E1"],
        [0.75, "#4082C0"], [1.0, "#0F65A0"],
    ]
    z = mat.to_numpy(dtype=float)
    text_colors = np.where(z > 0.6, "white", "#1A1A1A")

    fig = go.Figure(data=go.Heatmap(
        z=z, x=cols, y=cols,
        colorscale=colorscale, zmin=0, zmax=1,
        colorbar=dict(title="Cramér's V"),
        hovertemplate="%{x} ↔ %{y}<br>Cramér's V: %{z}<extra></extra>",
        hoverongaps=False,
    ))
    annotations = []
    for yi in range(len(cols)):
        for xi in range(len(cols)):
            v = mat.iat[yi, xi]
            if v != v:
                continue
            annotations.append(dict(
                x=cols[xi], y=cols[yi], text=f"{v:.3g}",
                showarrow=False, font=dict(color=text_colors[yi, xi], size=12),
            ))
    fig.update_layout(
        title="Cramér's V (categorical pairs)",
        height=max(450, 55 * len(cols) + 200),
        template="plotly_white",
        xaxis=dict(tickangle=-45),
        annotations=annotations,
        plot_bgcolor="#A1ACBD",  # NaN cells appear neutral grey
    )
    fig.update_yaxes(autorange="reversed")
    return fig.to_html(full_html=False, include_plotlyjs=False)


def eta_heatmap(df: pd.DataFrame) -> str | None:
    types = _classify_columns(df)
    cats = [c for c, t in types.items() if t == 'categorical']
    nums = [c for c, t in types.items() if t == 'numeric']
    if not cats or not nums:
        return None

    mat = pd.DataFrame(np.zeros((len(cats), len(nums))), index=cats, columns=nums)
    for c in cats:
        for nu in nums:
            val = _eta_squared(df, c, nu)
            mat.at[c, nu] = round(float(val), 3) if val == val else np.nan

    # light -> magenta (#994564)
    colorscale = [
        [0.0, "#F2F2F2"], [0.25, "#D27897"], [0.5, "#A83665"],
        [0.75, "#932263"], [1.0, "#994564"],
    ]
    z = mat.to_numpy(dtype=float)
    text_colors = np.where(z > 0.5, "white", "#1A1A1A")

    fig = go.Figure(data=go.Heatmap(
        z=z, x=nums, y=cats,
        colorscale=colorscale, zmin=0, zmax=1,
        colorbar=dict(title="Eta²"),
        hovertemplate="%{y} → %{x}<br>Eta²: %{z}<extra></extra>",
        hoverongaps=False,
    ))
    annotations = []
    for yi in range(len(cats)):
        for xi in range(len(nums)):
            v = mat.iat[yi, xi]
            if v != v:
                continue
            annotations.append(dict(
                x=nums[xi], y=cats[yi], text=f"{v:.3g}",
                showarrow=False, font=dict(color=text_colors[yi, xi], size=12),
            ))
    fig.update_layout(
        title="Eta² — variance in numeric explained by categorical (directional)",
        height=max(450, 55 * len(cats) + 200),
        template="plotly_white",
        xaxis_title="Numeric", yaxis_title="Categorical",
        xaxis=dict(tickangle=-45),
        annotations=annotations,
        plot_bgcolor="#A1ACBD",
    )
    fig.update_yaxes(autorange="reversed")
    return fig.to_html(full_html=False, include_plotlyjs=False)

def top_associations(values, methods, threshold=0.7):
    pairs = []
    cols = list(values.columns)
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            v = values.iat[i, j]
            if v == v and v >= threshold:
                pairs.append({
                    "var1": cols[i],
                    "var2": cols[j],
                    "value": round(float(v), 3),
                    "method": methods.iat[i, j],
                })
    return sorted(pairs, key=lambda p: p["value"], reverse=True) or None

"""

Missingness

"""
def littles_mcar_test(df: pd.DataFrame):
    return {"Hello":"World"}

def missing_matrix(df: pd.DataFrame):
    missing_values = df.isnull().astype(int)
    fig = px.imshow(missing_values, labels=dict(color="Missing Values"),
                    aspect="auto", color_continuous_scale="blues_r",
                    title="Missing Values Matrix")
    fig.update_yaxes(autorange='reversed')
    fig.update_layout(coloraxis_showscale=False)
    return fig.to_html(full_html=False, include_plotlyjs=False)


def missing_values_barchart(df: pd.DataFrame):
    missing_counts = df.isna().sum()
    fig = px.bar(x=missing_counts.index, y=missing_counts.values,
                 labels={'x': 'Columns', 'y': 'Missing Values'},
                 color_discrete_sequence=['#0F65A0'])
    fig.update_layout(title="Missing Values per Column", bargap=0.2, plot_bgcolor='white')
    apply_standard_axes(fig, tick_angle=-45)
    return fig.to_html(full_html=False, include_plotlyjs=False)


def balance_plot(df, target):
    fig = px.histogram(df, x=target, color_discrete_sequence=["#0F65A0"], text_auto=True)
    fig.update_layout(title="Class Balance (Target Distribution)", bargap=0.2, plot_bgcolor="white")
    apply_standard_axes(fig)
    return fig.to_html(full_html=False, include_plotlyjs=False)


def boxplot(df: pd.DataFrame):
    df = df.select_dtypes(include=['float64', 'int64']).dropna(axis=1, how='all')
    fig = go.Figure()
    for col in df:
        fig.add_trace(go.Box(y=df[col].values, name=df[col].name))
    fig.update_yaxes(type="log", title="Logarithmic", showticklabels=False)
    fig.update_layout(
        xaxis=dict(rangeslider=dict(visible=True), type="linear"),
        title="Boxplot", xaxis_title="Columns",
        yaxis_title="Values", legend_title="Columns",
    )
    return fig.to_html(full_html=False, include_plotlyjs=False)


def scatter_matrix(df: pd.DataFrame):
    df = df.select_dtypes(include=['float64', 'int64']).dropna(axis=1, how='all')
    fig = px.scatter_matrix(df, color_discrete_sequence=["#0F65A0"], height=750)
    fig.update_traces(diagonal_visible=False,
                      marker=dict(size=2, opacity=0.5, color="#0F65A0"))
    fig.update_layout(title="Scatter Matrix", xaxis_title="Columns",
                      plot_bgcolor="white", bargap=0.2, dragmode="select")

    n_vars = len(df.columns)
    for i in range(1, n_vars + 1):
        for j in range(1, n_vars + 1):
            if i == 1:
                fig.update_layout({f'xaxis{j if j > 1 else ""}': dict(
                    mirror=True, ticks="outside", showline=True,
                    linecolor="black", linewidth=1, gridcolor="lightgrey",
                    title_standoff=25,
                )})
            if j == 1:
                fig.update_layout({f'yaxis{i if i > 1 else ""}': dict(
                    mirror=True, ticks="outside", showline=True,
                    linecolor="black", linewidth=1, gridcolor="lightgrey",
                    title_standoff=25,
                )})

    fig.for_each_annotation(lambda a: a.update(
        textangle=0,
        x=-0.08 if a.textangle == 90 or 'y' in str(a.yref) else a.x,
        y=-0.08 if a.textangle == 0 and 'x' in str(a.xref) else a.y,
    ))
    return fig.to_html(full_html=False, include_plotlyjs=False)