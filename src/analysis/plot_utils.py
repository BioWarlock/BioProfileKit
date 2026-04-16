import plotly.express as px


def apply_standard_axes(fig, tick_angle=None):
    x_kwargs = dict(mirror=True, ticks='outside', showline=True,
                    linecolor='black', gridcolor='lightgrey')
    if tick_angle is not None:
        x_kwargs['tickangle'] = tick_angle
    fig.update_xaxes(**x_kwargs)
    fig.update_yaxes(mirror=True, ticks='outside', showline=True,
                     linecolor='black', gridcolor='lightgrey')


def plot_overview(col):
    if col.dtype != 'object':
        bins = None if col.nunique() < 10 else 10
        fig = px.histogram(x=col, color_discrete_sequence=['#0F65A0'])
        fig.update_layout(bargap=0.2, plot_bgcolor='white')
        apply_standard_axes(fig)
        return fig.to_html(full_html=False, include_plotlyjs=False)
    return None