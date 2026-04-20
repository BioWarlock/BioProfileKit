import plotly.express as px


def apply_standard_axes(fig, tick_angle=None):
    x_kwargs = dict(mirror=True, ticks='outside', showline=True,
                    linecolor='black', gridcolor='lightgrey')
    if tick_angle is not None:
        x_kwargs['tickangle'] = tick_angle
    fig.update_xaxes(**x_kwargs)
    fig.update_yaxes(mirror=True, ticks='outside', showline=True,
                     linecolor='black', gridcolor='lightgrey')

#ToDo: Check if we want to keep nan in count plot.
def plot_overview(col):
    col = col.dropna()
    if col.dtype in ('str', 'string'):
        truncated = col.apply(lambda x: str(x)[:20] + '…' if len(str(x)) > 20 else str(x))
    if col.dtype != 'object':
        bins = col.nunique() if col.nunique() < 20 else 20
        if col.dtype in ('str', 'string'):
            fig = px.histogram(x=truncated, color_discrete_sequence=['#0F65A0'])
            apply_standard_axes(fig, tick_angle=-45)
        else:
            fig = px.histogram(x=col, color_discrete_sequence=['#0F65A0'], nbins=bins)
            apply_standard_axes(fig)
        fig.update_layout(bargap=0.4, plot_bgcolor='white', xaxis_title=col.name,yaxis_title='Count')
        fig.layout.xaxis.automargin = True
        return fig.to_html(full_html=False, include_plotlyjs=False)
    return None