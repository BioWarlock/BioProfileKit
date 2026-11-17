import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import base64
import gzip
import uuid
import plotly.io as pio


MAX_CATEGORIES = 1000

def fig_html(fig, config: dict | None = None) -> str:
    spec = fig.to_plotly_json()                 # {"data": [...], "layout": {...}}
    if config:
        spec["config"] = config
    js = pio.json.to_json_plotly(spec)          # numpy -> bdata wie bei fig.to_json()
    div_id = f"bpk-{uuid.uuid4().hex[:12]}"
    payload = base64.b64encode(gzip.compress(js.encode("utf-8"), 9)).decode("ascii")
    return (f'<div id="{div_id}" class="plotly-graph-div"></div>'
            f'<script>bpkPlot("{div_id}", "{payload}");</script>')

def _config(filename: str, width: int = 1600, height: int = 1200) -> dict:
    return {
        'toImageButtonOptions': {
            'format': 'png',   # one of png, svg, jpeg, webp
            'filename': filename,
            'height': height,
            'width': width,
            'scale': 4,
        }
    }

def apply_standard_axes(fig, tick_angle=None):
    x_kwargs = dict(mirror=True, ticks='outside', showline=True,
                    linecolor='black', gridcolor='lightgrey')
    if tick_angle is not None:
        x_kwargs['tickangle'] = tick_angle
    fig.update_xaxes(**x_kwargs)
    fig.update_yaxes(mirror=True, ticks='outside', showline=True,
                     linecolor='black', gridcolor='lightgrey')
def _compact(a) -> np.ndarray:
    a = np.asarray(a)
    if a.dtype == np.float64:
        f32 = a.astype(np.float32)
        if np.array_equal(f32.astype(np.float64), a, equal_nan=True):
            a = f32
    if a.dtype.kind == "f" and a.size and np.all(np.isfinite(a)) and np.all(a == np.round(a)):
        a = a.astype(np.int64)
    if a.dtype.kind in "iu" and a.size:
        a = a.astype(np.result_type(np.min_scalar_type(a.min()), np.min_scalar_type(a.max())))
    return a


def count_histogram(series: pd.Series, **px_kwargs):
    name = series.name or "value"
    vc = series.dropna().value_counts(sort=False)
    x = vc.index.to_numpy()
    if x.dtype != object:
        x = _compact(x.astype(float))
    counts = pd.DataFrame({name: x, "count": _compact(vc.to_numpy())})
    fig = px.histogram(counts, x=name, y="count", histfunc="sum", **px_kwargs)
    fig.update_traces(hovertemplate=f"{name}=%{{x}}<br>count=%{{y}}<extra></extra>")
    fig.update_yaxes(title_text="count")
    return fig

#ToDo: Check if we want to keep nan in count plot.
def _truncate(x, n: int = 20) -> str:
    s = str(x)
    return s[:n] + '…' if len(s) > n else s

import plotly.graph_objects as go

def count_bars_gl(series: pd.Series, color: str = '#0F65A0'):
    vc = series.value_counts(sort=False)      # Reihenfolge = erstes Auftreten, wie px.histogram
    labels = vc.index.to_numpy(dtype=object)
    counts = vc.to_numpy()
    n = len(vc)

    # Jeder Balken als senkrechte Linie 0 -> count, getrennt durch NaN
    y = np.empty(3 * n, dtype=np.float32)
    y[0::3], y[1::3], y[2::3] = 0, counts, np.nan

    fig = go.Figure(go.Scattergl(
        x=np.repeat(labels, 3), y=y, mode='lines',
        line=dict(color=color, width=1), hoverinfo='skip', showlegend=False))
    fig.update_xaxes(type='category')
    fig.update_yaxes(range=[0, counts.max() * 1.05])
    fig.update_layout(hovermode=False)
    return fig

def plot_overview(col: pd.Series):
    col = col.dropna()
    nunique = col.nunique()
    if nunique == 1:
        return " "  # ToDo find better solution
    if col.dtype == 'object':
        return None

    if col.dtype in ('str', 'string'):
        truncated = col.map(_truncate)
        if nunique > MAX_CATEGORIES:
            fig = count_bars_gl(truncated)
        else:
            fig = count_histogram(truncated, color_discrete_sequence=['#0F65A0'])
        apply_standard_axes(fig, tick_angle=-45)
        x_title = "Rank (by frequency)" if nunique > MAX_CATEGORIES else col.name
    else:
        fig = count_histogram(col, color_discrete_sequence=['#0F65A0'],
                              nbins=min(nunique, 20))
        apply_standard_axes(fig)
        x_title = col.name
    title = f"Value Distribution – {col.name}"
    if nunique > MAX_CATEGORIES:
        title += f" ({nunique:,} distinct values)"
    fig.update_layout(title=title, bargap=0.4, plot_bgcolor='white', xaxis_title=x_title, yaxis_title='Count', autosize=True)
    fig.layout.xaxis.automargin = True
    return fig_html(fig, config=_config(f"{col.name}_overview"))