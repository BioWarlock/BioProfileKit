from pandas import DataFrame
from plotly import express as px, graph_objs as go, graph_objects as go
from plotly.subplots import make_subplots

config = {
  'toImageButtonOptions': {
    'format': 'png', # one of png, svg, jpeg, webp
    'filename': None,
    'height': 1200,
    'width': 1600,
    'scale': 4
  }
}


def length_distribution(all_overview: DataFrame, unit: str = "bp"):
    bins = 300 if all_overview.shape[0] > 10000 else all_overview.shape[0]
    fig = px.histogram(
        all_overview,
        x="lengths",
        title="Sequence Length Distribution Across All Sequences",
        labels={"lengths": f"Sequence Length ({unit})"},
        color_discrete_sequence=['#0F65A0'],
        nbins=bins
    )
    fig.update_traces(
        marker_line_color="#616D78",
        marker_line_width=1,
        opacity=0.85
    )
    fig.update_layout(
        template="plotly_white",
        xaxis_title=f"Sequence Length ({unit})",
        yaxis_title="Sequence Count",
        bargap=0.5
    )
    config['toImageButtonOptions']['filename'] = "length_distribution"
    return fig.to_html(full_html=False, include_plotlyjs=False, config=config)

def gc_distribution(all_overview: DataFrame):
    fig = px.violin(
        all_overview,
        y="gc_content",
        box=True,
        points='all',
        title="Distribution of GC Content Across All Sequences",
        labels={"gc_content": "GC Content (%)"},
    )
    fig.update_traces(
        fillcolor="#AB7E8F",
        line_color="#616D78",
        box_fillcolor="#994564",
        box_line_color="#616D78",
        marker=dict(
            color="#0F65A0",
            size=3,
        ),
        opacity=0.85,
        line_width=1,
    )
    fig.update_layout(
        template="plotly_white",
        yaxis=dict(
            range=[0, 100],
            ticksuffix="%"
        )
    )
    config['toImageButtonOptions']['filename'] = "gc_distribution"
    return fig.to_html(full_html=False, include_plotlyjs=False, config=config)

def ambiguous_distribution(all_overview: DataFrame, col: str = "N", label: str = "N-Count"):
    fig = go.Figure()

    fig.add_trace(go.Scattergl(
        x=all_overview.index,
        y=all_overview[col],  # .sort_values(ascending=False),
        mode='lines',
        line=dict(color='#994564', width=0.85),
        name=label
    ))

    fig.update_layout(
        title=f"Complete Distribution Curve of {label}s Across All Sequences",
        xaxis_title="Sequence Index (Chronological Order)",
        yaxis_title=f"Number of '{label}' in Sequence",
        template="plotly_white",
        yaxis=dict(rangemode="tozero")
    )

    return fig.to_html(full_html=False, include_plotlyjs=False)


def at_gc_skewness(all_overview: DataFrame):
    fig = make_subplots(
        rows=2, cols=2,
        row_heights=[0.12, 0.88],
        column_widths=[0.88, 0.12],
        shared_xaxes=True,
        shared_yaxes=True,
        horizontal_spacing=0.02,
        vertical_spacing=0.02
    )

    fig.add_trace(
        go.Histogram(
            x=all_overview['GC_skew'], marker=dict(
                color='#0F65A0',
                line=dict(
                    color='#616D78',
                    width=0.5
                )
            ),
            name='GC Skew Dist.',
            hovertemplate='GC Skew: %{x}<br>Count: %{y}',
            showlegend=False,
        ),
        row=1, col=1
    )

    hover_labels = []
    for idx, seq in zip(all_overview.index, all_overview['sequence']):
        if isinstance(seq, str) and seq.strip():
            short_seq = f"{seq[:20]}..." if len(seq) > 20 else seq
            hover_labels.append(f"#{idx}: {short_seq}")
        else:
            hover_labels.append(f"#{idx}: [No Sequence Data]")

    all_overview['hover_sequence_label'] = hover_labels

    fig.add_trace(
        go.Scatter(
            x=all_overview['GC_skew'],
            y=all_overview['AT_skew'],
            mode='markers',
            marker=dict(color='#994564', opacity=0.3),
            customdata=all_overview[['hover_sequence_label']],
            name='%{customdata[0]}',
            hovertemplate='<b>Seq:</b> %{customdata[0]}<br>GC Skew: %{x}<br>AT Skew: %{y}<extra></extra>',
            showlegend=False
        ),
        row=2, col=1
    )

    fig.add_trace(
        go.Histogram(
            y=all_overview['AT_skew'],
            marker=dict(
                color='#0F65A0',
                line=dict(
                    color='#616D78',
                    width=0.5
                )
            ),
            name='AT Skew Dist.',
            hovertemplate='AT Skew: %{y}<br>Count: %{x}',
            showlegend=False,

        ),
        row=2, col=2
    )

    fig.update_layout(
        title='Skewness Distribution',
        template='plotly_white',
        bargap=0.05
    )

    fig.update_xaxes(title_text="GC Skew (G-C)/(G+C)", range=[-1.05, 1.05], row=2, col=1)
    fig.update_yaxes(title_text="AT Skew (A-T)/(A+T)", range=[-1.05, 1.05], row=2, col=1)

    fig.add_hline(y=0, line_dash="dash", line_color="rgba(0,0,0,0.2)", row=2, col=1)
    fig.add_vline(x=0, line_dash="dash", line_color="rgba(0,0,0,0.2)", row=2, col=1)
    config['toImageButtonOptions']['filename'] = "at_gc_skewness"
    return fig.to_html(full_html=False, include_plotlyjs=False, config=config)

def aa_group_distribution(group_dist: dict):
    groups = list(group_dist.keys())
    values = [round(v * 100, 2) for v in group_dist.values()]

    fig = px.bar(
        x=groups,
        y=values,
        title="Amino Acid Group Distribution Across All Sequences",
        labels={"x": "Amino Acid Group", "y": "Proportion (%)"},
        color_discrete_sequence=['#0F65A0'],
    )
    fig.update_traces(
        marker_line_color="#616D78",
        marker_line_width=1,
        opacity=0.85,
    )
    fig.update_layout(
        template="plotly_white",
        xaxis_title="Amino Acid Group",
        yaxis_title="Proportion (%)",
        bargap=0.3,
    )
    config['toImageButtonOptions']['filename'] = "aa_group_distribution"
    return fig.to_html(full_html=False, include_plotlyjs=False, config=config)