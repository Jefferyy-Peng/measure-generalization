import pandas as pd
import numpy as np
from dash import Dash, dcc, html, Input, Output
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.stats import norm, boxcox, yeojohnson

def transform(y_raw, y_transform):
    if y_transform == 'probit':
        y = norm.ppf(y_raw.clip(eps, 1 - eps))
    elif y_transform == 'boxcox':
        y = y_raw.clip(lower=eps)
        y, _ = boxcox(y)
    elif y_transform == 'yeojohnson':
        y = yeojohnson(y_raw.fillna(y_raw.mean()))[0]
    else:
        y = y_raw
    return y

# -------------- Parameters --------------
df = pd.read_csv("metrics/terra-incognita-46_model_location_100_kl_data.csv", comment="#")
# df = df[~df['model_id'].isin([6, 102, 200])]
target = 'test_f1_location_100'
y_transform = None  # Options: 'probit', 'boxcox', 'yeojohnson', or None
log_transform = False  # Whether to apply log to eligible x-axis features
log_include = {
    'circuit_instability',
    'normed_circuit_instability',
    'weighted_shortcut_score',
    'logit_contribution_diff_deep_vs_shallow',
    'edge_start_ratio_deep_vs_shallow',
    'edge_start_ratio_deep_vs_shallow_1',
    'edge_start_ratio_deep_vs_shallow_2',
    'edge_start_ratio_deep_vs_shallow_4',
    'edge_start_ratio_deep_vs_shallow_5',
    'logit_contribution_ratio_deep_vs_shallow',
    'logit_contribution_ratio_deep_vs_shallow_1',
    'logit_contribution_ratio_deep_vs_shallow_2',
    'logit_contribution_ratio_deep_vs_shallow_4',
    'logit_contribution_ratio_deep_vs_shallow_5',
    'deep_logit_contribution',
    'deep_logit_contribution_signed',
    'deep_logit_contribution_signed_normed',
    'algebraic_connectivity',
    'spectral_gap',
    'graph_energy',
    'community_size_variance',
    'avg_clustering',
    'kirchhoff_index',
    'avg_path_length',
    'path_redundancy',
    'effective_path_depth',
    'avg_betweenness',
    'spectral_radius',
    # 'normed_algebraic_connectivity',
    'normed_graph_energy',
    'normed_spectral_radius',
    'layerwise_score_variance',
    # 'test_EMD_location_100'
    # 'test_EMD_location_43'
}
plot_include = ['model_id', 'shortcut_vs_deep_ratio', 'edge_start_ratio_deep_vs_shallow_1', 'logit_contribution_ratio_deep_vs_shallow']
eps = 1e-16
# ----------------------------------------

# ----------- Y-axis transform ----------
# y_raw = (df["val_id_acc"] - df[target]).copy()
y_raw = (df[target]).copy()

y = transform(y_raw, y_transform)
# ----------------------------------------

# ----------- X-axis features -----------
features = [col for col in df.columns if (col != target and col in plot_include)]
cols = 3
rows = (len(features) + cols - 1) // cols

fig = make_subplots(
    rows=rows, cols=cols,
    shared_yaxes='all',
    subplot_titles=features,
    vertical_spacing=0.01,
    horizontal_spacing=0.08
)
fig.update_layout(font=dict(family="Arial", size=10))

for i, feat in enumerate(features):
    r, c = i // cols + 1, i % cols + 1
    x = df[feat]

    if feat == 'val_id_acc' or feat == 'test_id_acc':
        x = transform(x, y_transform)

    if 'srcc' in feat or 'kendal' in feat or 'pearson' in feat or 'jaccard' in feat or 'cosine' in feat:
        x = transform(x, y_transform)

    if feat == 'spectral_gap':
        x = abs(df[feat])

    if feat == 'path_redundancy':
        x += 1

    # Apply log transform only if enabled
    if log_transform:
        if (
            feat in log_include
            # and pd.api.types.is_numeric_dtype(x)
            # and (x > 0).all()
        ):
            x = np.log(x.clip(lower=eps)+1).fillna(0)

    fig.add_trace(
        go.Scatter(
            x=x, y=y,
            mode='markers', name=feat,
            marker=dict(size=6)
        ), row=r, col=c
    )
    fig.update_xaxes(title_text=feat, row=r, col=c)

fig.update_yaxes(title_text="Worst Group Accuracy", row=rows, col=1)
fig.update_layout(
    height=400 * rows,
    width=420 * cols,
    dragmode='select',
    hovermode='closest',
    uirevision='keep_selection'
)

# --------- Dash app setup ------------
app = Dash(__name__)
app.layout = html.Div([dcc.Graph(id="subplot-graph", figure=fig)])

@app.callback(
    Output("subplot-graph", "figure"),
    Input("subplot-graph", "selectedData")
)
def highlight(selection):
    fig2 = fig
    if selection and selection.get("points"):
        idxs = {pt["pointIndex"] for pt in selection["points"]}
        fig2.update_traces(
            selectedpoints=list(idxs),
            selected=dict(marker=dict(color='red', opacity=1)),
            unselected=dict(marker=dict(color='blue', opacity=0.2))
        )
    return fig2

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8051)
