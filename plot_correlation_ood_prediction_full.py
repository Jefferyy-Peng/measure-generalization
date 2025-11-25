import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold
from scipy.stats import norm, boxcox, yeojohnson, spearmanr, kendalltau

# ==== Transform function (same as your code) ====
eps = 1e-16
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

def safe_spearman(x, y):
    try:
        rho, p = spearmanr(x, y, nan_policy='omit')
    except Exception:
        rho, p = np.nan, np.nan
    # spearmanr returns nan if constant; coerce to 0 if you prefer:
    if np.isnan(rho):
        rho = 0.0
    return rho

def safe_kendall(x, y):
    """
    Compute Kendall Rank Correlation Coefficient (Kendall's tau).
    Handles edge cases (constant vectors, NaNs) gracefully.
    """
    try:
        tau, _ = kendalltau(x, y, nan_policy='omit')
    except Exception:
        tau = np.nan
    return 0.0 if np.isnan(tau) else tau

# ==== Config ====
TOPN = "_200"
tasks = ['PACS-set2', 'fmow-set2', 'camelyon17-set2']
y_transform = None   # 'probit', 'boxcox', 'yeojohnson', or None
# df = pd.read_csv(f"metrics/{task}_model_1_kl_resid{TOPN}_data.csv", comment="#")
# df["acc_raw"] = df["acc"]
# Choose target variable
target_col = "f1"
# df[target_col] = transform(df[target_col], y_transform)

# Transform baselines too
# for bcol in ["AC"]:
#     if bcol in df.columns:
#         df[bcol] = transform(df[bcol], y_transform)

# Select circuit metrics (drop non-metrics cols)
non_metric_cols = {"domain", "acc", "f1", "AC", "ANE", "EMD", "l2", "model_id"}
# circuit_metrics = [c for c in df.columns if c not in non_metric_cols]
circuit_metrics = [
    # "normed_spectral_radius",
    "abs_srcc",
    # "abs_kendall_tau_c",
    # "abs_pearson",
    "abs_cosine",
    "abs_residual_norm_L2",
    # "trim_rbo",
    # "weighted_jaccard",
    # "layer_pearson",
    # "layer_spearman",
    "binary_jaccard",
    "laplacian_spectral_dist",
    "netlsd_dist",
    # "motif_counts_dist",
    # "degree_distribution_dist",
    # "weighted_path_depth",
    # "layerwise_score_entropy",
    # "shortcut_reliance",
    # "layerwise_score_variance",
    # "normed_graph_energy",
    # "weighted_shortcut_score",
    # "circuit_instability"
]

# (Optional) log-transform some metrics
log_include = {
    'circuit_instability',
    "normed_spectral_radius",
    'logit_contribution_ratio_deep_vs_shallow',
    'spectral_gap',
    'avg_path_length',
    'effective_path_depth',
    'spectral_radius',
    # "abs_srcc",
    'weighted_path_depth',
}
eps = 1e-16
# for col in log_include:
#     if col in df.columns:
#         df[col] = np.log(df[col].clip(lower=eps) + 20)

metric_name = {
    "abs_srcc": r"$\mathrm{CSS}_{(v,\text{SRCC})}$",
    "binary_jaccard": r'$\mathrm{CSS}_{(g,\text{Jaccard})}$',
    "laplacian_spectral_dist": r'$\mathrm{CSS}_{(g,\text{Laplacian})}$',
    "netlsd_dist": r'$\mathrm{CSS}_{(g,\text{NetLSD})}$',
    "abs_cosine": r'$\mathrm{CSS}_{(v,\text{cosine})}$',
    "abs_residual_norm_L2": r'$\mathrm{CSS}_{(v,\ell^2)}$',
}

# ---- Plotting Setup ----
baseline_cols = ["AC", "ANE", "EMD", "rankme", "alphaReQ","ATC", ]
# baseline_cols = ["AC", "ANE", "ATC", "EMD", "alphaReQ"]

use_multivar = False  # set True if you un-comment the multivariate block
extra_rows = (1 if use_multivar else 0) + len(baseline_cols)
ncols = len(circuit_metrics) + extra_rows
nrows = len(tasks)
fig, axes = plt.subplots(
    nrows=nrows, ncols=ncols,
    figsize=(3*ncols, 3*nrows),
    gridspec_kw=dict(wspace=0.0, hspace=0.5)   # <- no gap across a row, tighter between rows
)

y_label = {
    'acc': 'OOD Accuracy',
    'f1': 'OOD F1'
}

for row_idx, task in enumerate(tasks):
    if 'camelyon17' in task:
        df = pd.read_csv(f"metrics/{task}_model_0_kl{TOPN}_data.csv", comment="#")
    else:
        df = pd.read_csv(f"metrics/{task}_model_0_kl{TOPN}_data.csv", comment="#")
    # df["acc_raw"] = df["acc"]
    if task == 'PACS-set2':
        target_col = 'acc'
    elif task == 'fmow-set2':
        target_col = 'f1'
    elif task == 'camelyon17-set2':
        target_col = 'f1'
    df[target_col] = transform(df[target_col], y_transform)

    # log transforms per task
    for col in log_include:
        if col in df.columns:
            df[col] = np.log(df[col].clip(lower=eps))
        # if col == 'abs_srcc':
        #     df[col] = transform(df[col])

    # 1) univariate (rows 0..len(circuit_metrics)-1)
    for col_idx, metric in enumerate(circuit_metrics):
        ax = axes[row_idx, -(col_idx + 1)]
        x = df[metric].values.reshape(-1, 1)
        y = df[target_col].values
        reg = LinearRegression().fit(x, y)
        y_pred = reg.predict(x)
        r2 = r2_score(y, y_pred)
        rho = safe_spearman(x[:, 0], y)
        tau = safe_kendall(x[:, 0], y)
        ax.scatter(x, y, c='tab:orange', marker='^',alpha=0.6)
        ax.plot(x, y_pred, color="red")
        ax.text(0.05, 0.95, f"$R^2$={r2:.3f}\nSRCC={rho:.3f}\nKRCC={tau:.3f}", transform=ax.transAxes, va='top',
                ha='left', fontsize=10,
                bbox=dict(boxstyle="round,pad=0.3", edgecolor='gray', facecolor='white', alpha=0.8))
        # if row_idx == 0:
        #     ax.set_title(task)
        ax.set_xlabel(metric_name[metric])

    # 2) (optional) multivariate block
    # if use_multivar:
    #     row = len(circuit_metrics)
    #     ax = axes[row, col_idx]
    #     ...

    # 3) baselines (rows start after circuit metrics [+1 if multivar])
    base_col_start = 0
    for i, bcol in enumerate(baseline_cols):
        ax = axes[row_idx, base_col_start + i]
        if bcol not in df.columns: continue
        x = df[bcol].values.reshape(-1, 1)
        y = df[target_col].values
        reg = LinearRegression().fit(x, y)
        y_pred = reg.predict(x)
        r2 = r2_score(y, y_pred)
        rho = safe_spearman(x[:, 0], y)
        tau = safe_kendall(x[:, 0], y)
        ax.scatter(x, y, c='tab:orange', marker='^', alpha=0.6)
        order = np.argsort(x[:, 0])
        ax.plot(x[order], y_pred[order], color="red")
        ax.text(0.05, 0.95, f"$R^2$={r2:.3f}\nSRCC={rho:.3f}\nKRCC={tau:.3f}", transform=ax.transAxes, va='top',
                ha='left', fontsize=10,
                bbox=dict(boxstyle="round,pad=0.3", edgecolor='gray', facecolor='white', alpha=0.8))
        ax.set_xlabel(bcol)
        if i == 0:
            ax.set_ylabel(y_label[target_col])


def add_row_titles_between(fig, axes, titles, pad_top=0.01, **textkw):
    """
    Place one title above row 0, then between each subsequent pair of rows.
    Works even if axes share x/y or have zero hspace.
    """
    # if axes are transposed (shape = ncols x nrows), fix it
    if axes.shape[0] != len(titles):
        axes = axes.T

    fig.canvas.draw()  # ensure positions are up-to-date
    nrows, ncols = axes.shape

    # collect per-row bottoms and tops from the full row (all columns)
    row_bottom = []
    row_top = []
    for r in range(nrows):
        y0s = [axes[r, c].get_position().y0 for c in range(ncols)]
        y1s = [axes[r, c].get_position().y1 for c in range(ncols)]
        row_bottom.append(min(y0s))
        row_top.append(max(y1s))

    # place title above first row
    y = row_top[0] + pad_top
    fig.text(0.5, y, titles[0], ha='center', va='bottom', **textkw)

    # place titles between subsequent rows
    for r in range(1, nrows):
        y = 0.5 * (row_bottom[r-1] + row_top[r])
        fig.text(0.5, y, titles[r], ha='center', va='center', **textkw)

row_titles = [
    r'PACS',
    r'FMoW',
    r'Camelyon17',
]

# share y within each row and hide repeated y labels
for r in range(nrows):
    for c in range(1, ncols):
        axes[r, c].sharey(axes[r, 0])
        axes[r, c].tick_params(labelleft=False)  # keep y labels only on first column


for axrow, title in zip(axes, row_titles):
    y = axrow[0].get_position().y1 + 0.02   # a little above the row
    fig.text(0.5, y, title, ha='center', va='bottom',
             fontsize=14, fontweight='bold')

for ax in axes.flat:
    ax.set_box_aspect(0.8)
    ax.tick_params(axis='both', which='major', labelsize=8)

plt.savefig(f"figures/domain_fit_plots{TOPN}_new_full_1.pdf", transparent=True,
            bbox_inches="tight",   # trim empty margins
            pad_inches=0.01)
# plt.show()
