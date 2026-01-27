import plotly.express as px
import pandas as pd
import plotly.graph_objects as go
from dataclasses import dataclass

@dataclass
class GeneralPlots:
    correlation_heatmap: str
    missing_matrix: str
    missing_values_barchart: str
    balance_plot: str | None
    boxplot: str
    scatter_matrix: str

def general_plots(df: pd.DataFrame, target: str) -> GeneralPlots:
    return GeneralPlots(
        correlation_heatmap=correlation_heatmap(df),
        missing_matrix=missing_matrix(df),
        missing_values_barchart=missing_values_barchart(df),
        balance_plot=balance_plot(df, target) if target else None,
        boxplot=boxplot(df),
        scatter_matrix=scatter_matrix(df)
    )

def correlation_heatmap(df: pd.DataFrame):
    df_numeric = df.select_dtypes(include=['float64', 'int64']).dropna(axis=1, how='all')
    if not df_numeric.empty:
        std = df_numeric.std(ddof=0)
        df_numeric = df_numeric.loc[:, std[std > 0].index]
    corr_matrix = df_numeric.corr()
    corr_matrix = round(corr_matrix, 3)
    fig = px.imshow(corr_matrix, text_auto=True, labels=dict(color="Correlation"), color_continuous_scale="RdBu_r", aspect="auto", height=700)
    fig.update_layout(title="Correlation Heatmap")
    return fig.to_html(full_html=False, include_plotlyjs='cdn')


def missing_matrix(df: pd.DataFrame):
    missing_values = df.isnull().astype(int)

    fig = px.imshow(
        missing_values,
        labels=dict(color="Missing Values"),
        aspect="auto",
        color_continuous_scale="blues_r",
        title="Missing Values Matrix"
    )
    fig.update_yaxes(autorange='reversed')
    fig.update_layout(coloraxis_showscale=False)


    return fig.to_html(full_html=False, include_plotlyjs='cdn')

def missing_values_barchart(df: pd.DataFrame):
    missing_counts = df.isna().sum()
    fig = px.bar(x=missing_counts.index, y=missing_counts.values, labels={'x': 'Columns', 'y': 'Missing Values'},color_discrete_sequence=['#0F65A0'])
    fig.update_layout(title="Missing Values per Column", bargap=0.2, plot_bgcolor='white')
    fig.update_xaxes(
        tickangle=-45,
        mirror=True,
        ticks='outside',
        showline=True,
        linecolor='black',
        gridcolor='lightgrey'
    )
    fig.update_yaxes(
        mirror=True,
        ticks='outside',
        showline=True,
        linecolor='black',
        gridcolor='lightgrey'
    )

    return fig.to_html(full_html=False, include_plotlyjs='cdn')

def balance_plot(df, target):
    fig = px.histogram(df, x=target, color_discrete_sequence=["#0F65A0"], text_auto=True)
    fig.update_layout(
        title="Class Balance (Target Distribution)",
        bargap=0.2,
        plot_bgcolor="white"
    )
    fig.update_xaxes(
        mirror=True,
        ticks="outside",
        showline=True,
        linecolor="black",
        gridcolor="lightgrey"
    )
    fig.update_yaxes(
        mirror=True,
        ticks="outside",
        showline=True,
        linecolor="black",
        gridcolor="lightgrey"
    )
    return fig.to_html(full_html=False, include_plotlyjs='cdn')

def boxplot(df: pd.DataFrame):
    df = df.select_dtypes(include=['float64', 'int64'])
    fig = go.Figure()

    for col in df:
        fig.add_trace(go.Box(y=df[col].values, name=df[col].name))
    fig.update_yaxes(type="log", title="Logarithmic",showticklabels=False)
    fig.update_layout(
        title="Boxplot",
        xaxis_title="Columns",
        yaxis_title="Values",
        legend_title="Columns"
    )

    return fig.to_html(full_html=False, include_plotlyjs='cdn')

def scatter_matrix(df: pd.DataFrame):
    df = df.select_dtypes(include=['float64', 'int64'])
    fig = px.scatter_matrix(df, color_discrete_sequence=["#0F65A0"], height=750)

    fig.update_traces(
        diagonal_visible=False,
        marker=dict(size=2, opacity=0.5, color="#0F65A0")
    )

    # Layout anpassen
    fig.update_layout(
        title="Scatter Matrix",
        xaxis_title="Columns",
        plot_bgcolor="white",
        bargap=0.2,
        dragmode="select"
    )
    n_vars = len(df.columns)
    for i in range(1, n_vars + 1):
        for j in range(1, n_vars + 1):
            if i == 1:
                fig.update_layout({f'xaxis{j if j > 1 else ""}': dict(
                    mirror=True,
                    ticks="outside",
                    showline=True,
                    linecolor="black",
                    linewidth=1,
                    gridcolor="lightgrey",
                    title_standoff=25
                )})
            if j == 1:
                fig.update_layout({f'yaxis{i if i > 1 else ""}': dict(
                    mirror=True,
                    ticks="outside",
                    showline=True,
                    linecolor="black",
                    linewidth=1,
                    gridcolor="lightgrey",
                    title_standoff=25
                )})
    fig.for_each_annotation(lambda a: a.update(
        textangle=0,
        x=-0.08 if a.textangle == 90 or 'y' in str(a.yref) else a.x,
        y=-0.08 if a.textangle == 0 and 'x' in str(a.xref) else a.y
    ))

    return fig.to_html(full_html=False, include_plotlyjs='cdn')
