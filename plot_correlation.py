import os

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from scipy.stats import norm, boxcox, yeojohnson
from scipy.stats import spearmanr, kendalltau

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

def get_ID_OOD_name(task, dist):
    ID = task.split('-')[-1]
    if ID == 'PACS':
        ID = 'sketch'
    elif ID == 'camelyon17':
        ID = 'ID'
    elif ID == '38' or ID == '43' or ID == '46' or ID == '100':
        ID = 'location ' + ID

    ID = ID.replace('_', ' ')
    OOD = dist.replace('_', ' ')
    return ID, OOD

datasets = {
    # 'PACS': ['photo', 'cartoon', 'art_painting'],
    # 'PACS-photo': ['sketch', 'cartoon', 'art_painting'],
    # 'PACS-cartoon': ['photo', 'art_painting', 'sketch'],
    # 'PACS-art_painting' : ['photo', 'cartoon', 'sketch'],
    'camelyon17': ['hospital1', 'hospital2'],
    # 'metashift': ['ood'],
    # 'terra-incognita-38': ['location_43', 'location_46', 'location_100'],
    # 'terra-incognita-43': ['location_38', 'location_46', 'location_100'],
    # 'terra-incognita-46': ['location_38', 'location_43', 'location_100'],
    # 'terra-incognita-100': ['location_38', 'location_43', 'location_46']
}
# datasets = {
#     # 'PACS': ['photo', 'cartoon', 'art_painting'],
#     'PACS-photo': ['sketch'],
#     'camelyon17': ['hospital1'],
#     # 'metashift': ['ood'],
#     'terra-incognita-38': ['location_46']
# }
task_to_target = {
    'PACS': lambda d: 'val_id_acc' if d == 'id' else f'test_acc_{d}',
    'PACS-photo': lambda d: 'val_id_acc' if d == 'id' else f'test_acc_{d}',
    'PACS-cartoon': lambda d: 'val_id_acc' if d == 'id' else f'test_acc_{d}',
    'PACS-art_painting': lambda d: 'val_id_acc' if d == 'id' else f'test_acc_{d}',
    'camelyon17': lambda d: 'val_id_acc' if d == 'id' else f'test_acc_{d}',
    'metashift': lambda d: 'val_id_acc' if d == 'id' else f'test_id_acc',
    'terra-incognita-38': lambda d: 'val_id_f1' if d == 'id' else f'test_f1_{d}',
    'terra-incognita-43': lambda d: 'val_id_f1' if d == 'id' else f'test_f1_{d}',
    'terra-incognita-46': lambda d: 'val_id_f1' if d == 'id' else f'test_f1_{d}',
    'terra-incognita-100': lambda d: 'val_id_f1' if d == 'id' else f'test_f1_{d}',
}

baselines = {
    'PACS': [lambda d: 'sharpness', lambda d: f'test_AC_{d}', lambda d: f'test_ANE_{d}', lambda d: f'test_EMD_{d}', lambda d: f'test_rankme_{d}', lambda d: f'test_alphaReQ_{d}', lambda d: f'test_ATC_{d}',],
    'PACS-photo': [lambda d:'sharpness', lambda d: f'test_AC_{d}', lambda d: f'test_ANE_{d}', lambda d: f'test_EMD_{d}',  lambda d: f'test_rankme_{d}', lambda d: f'test_alphaReQ_{d}', lambda d: f'test_ATC_{d}'],
    'PACS-cartoon': [lambda d: 'sharpness', lambda d: f'test_AC_{d}', lambda d: f'test_ANE_{d}', lambda d: f'test_EMD_{d}', lambda d: f'test_rankme_{d}', lambda d: f'test_alphaReQ_{d}', lambda d: f'test_ATC_{d}'],
    'PACS-art_painting': [lambda d: 'sharpness', lambda d: f'test_AC_{d}', lambda d: f'test_ANE_{d}', lambda d: f'test_EMD_{d}', lambda d: f'test_rankme_{d}', lambda d: f'test_alphaReQ_{d}', lambda d: f'test_ATC_{d}'],
    'metashift': [lambda d: 'sharpness', lambda d: f'test_AC_{d}', lambda d: f'test_ANE_{d}', lambda d: f'test_EMD_{d}',  lambda d: f'test_rankme_{d}', lambda d: f'test_alphaReQ_{d}', lambda d: f'test_ATC_{d}'],
    'terra-incognita-38': [lambda d: 'sharpness', lambda d: f'test_AC_{d}', lambda d: f'test_ANE_{d}', lambda d: f'test_EMD_{d}',  lambda d: f'test_rankme_{d}', lambda d: f'test_alphaReQ_{d}', lambda d: f'test_ATC_{d}'],
    'terra-incognita-43': [lambda d: 'sharpness', lambda d: f'test_AC_{d}', lambda d: f'test_ANE_{d}',
                           lambda d: f'test_EMD_{d}', lambda d: f'test_rankme_{d}', lambda d: f'test_alphaReQ_{d}',
                           lambda d: f'test_ATC_{d}'],
    'terra-incognita-46': [lambda d: 'sharpness', lambda d: f'test_AC_{d}', lambda d: f'test_ANE_{d}',
                           lambda d: f'test_EMD_{d}', lambda d: f'test_rankme_{d}', lambda d: f'test_alphaReQ_{d}',
                           lambda d: f'test_ATC_{d}'],
    'terra-incognita-100': [lambda d: 'sharpness', lambda d: f'test_AC_{d}', lambda d: f'test_ANE_{d}',
                           lambda d: f'test_EMD_{d}', lambda d: f'test_rankme_{d}', lambda d: f'test_alphaReQ_{d}',
                           lambda d: f'test_ATC_{d}'],
    'camelyon17': [lambda d: 'sharpness', lambda d: f'test_AC_{d}', lambda d: f'test_ANE_{d}', lambda d: f'test_EMD_{d}',  lambda d: f'test_rankme_{d}', lambda d: f'test_alphaReQ_{d}', lambda d: f'test_ATC_{d}'],
}

# baselines = {
#     'PACS': [lambda d: f'test_AC_{d}', lambda d: f'test_EMD_{d}'],
#     'PACS-photo': [lambda d: f'test_AC_{d}', lambda d: f'test_EMD_{d}',],
#     # 'PACS-cartoon': [lambda d: f'test_AC_{d}', lambda d: f'test_EMD_{d}', lambda d: 'sharpness',
#     #                lambda d: f'test_ATC_{d}', lambda d: f'test_rankme_id', lambda d: f'test_alphaReQ_id',
#     #                lambda d: f'test_rankme_{d}', lambda d: f'test_alphaReQ_{d}'],
#     # 'metashift': [lambda d: f'test_AC_{d}', lambda d: f'test_EMD_{d}', lambda d:'sharpness', lambda d: f'test_ATC_{d}', lambda d: f'test_rankme_id', lambda d: f'test_alphaReQ_id', lambda d: f'test_rankme_{d}', lambda d: f'test_alphaReQ_{d}'],
#     'terra-incognita-38': [lambda d: f'test_AC_{d}', lambda d: f'test_EMD_{d}'],
#     'camelyon17': [lambda d: f'test_AC_{d}', lambda d: f'test_EMD_{d}'],
# }

circuit_type = '_kl'
y_transform = None
plot_include = [
    'logit_contribution_ratio_deep_vs_shallow',
    # 'edge_start_ratio_deep_vs_shallow',
    "shortcut_vs_deep_ratio",
    'edge_start_ratio_deep_vs_shallow_1',
    # 'edge_start_ratio_deep_vs_shallow_2',
    # 'edge_start_ratio_deep_vs_shallow_4',
    # 'edge_start_ratio_deep_vs_shallow_5',
    # 'shortcut_vs_local_ratio',
    # 'shortcut_vs_local_ratio_1',
    # 'shortcut_vs_local_ratio_2',
    # 'shortcut_vs_local_ratio_4',
    # 'shortcut_vs_local_ratio_5',
    # "shortcut_vs_deep_ratio_1",
    # "shortcut_vs_deep_ratio_2",
    # "shortcut_vs_deep_ratio_4",
    # "shortcut_vs_deep_ratio_5",
    # 'logit_contribution_ratio_deep_vs_shallow_1',
    # 'logit_contribution_ratio_deep_vs_shallow_2',
    # 'logit_contribution_ratio_deep_vs_shallow_4',
    # 'logit_contribution_ratio_deep_vs_shallow_5',
    # 'deep_logit_contribution'
    # 'circuit_instability',
    # 'deep_logit_contribution_normed',
    # 'spectral_gap',
    # 'modularity',
    # 'effective_path_depth',
    # 'spectral_radius',
    # 'algebraic_connectivity',
    # 'weighted_path_depth',
    # 'layerwise_score_entropy',
]
log_include = {
    'circuit_instability',
    "shortcut_vs_deep_ratio",
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
    'spectral_gap',
    'avg_path_length',
    'effective_path_depth',
    'spectral_radius',
    'weighted_path_depth',
    # 'layerwise_score_entropy'
    'EMD'
}

metric_name_map = {
    'logit_contribution_ratio_deep_vs_shallow': r'$\mathrm{DDB}_{\mathrm{out}}$ (ours)',
    'edge_start_ratio_deep_vs_shallow_1': r'$\mathrm{DDB}_{\mathrm{global}}$ (ours)',
    'shortcut_vs_deep_ratio': r'$\mathrm{DDB}_{\mathrm{deep}}$ (ours)',
    'val_id_acc': 'ID Accuracy',
}

task_name_map = {
    'PACS-photo-sketch': r'PACS (photo$\to$sketch)',
    'camelyon17-hospital1': r'Camelyon17 (ID$\to$Hospital1)',
    'terra-incognita-38-location_46': r'Terra Incognita (L38$\to$L46)',
}

task_2_title = {
    'PACS': 'PACS',
    'PACS-cartoon': 'PACS',
    'PACS-art_painting': 'PACS',
    'PACS-photo': 'PACS',
    'camelyon17': 'Camelyon17',
    'terra-incognita-38': 'Terra Incognita',
    'terra-incognita-43': 'Terra Incognita',
    'terra-incognita-46': 'Terra Incognita',
    'terra-incognita-100': 'Terra Incognita'
}

# --------- Plotting Setup ---------
ncols = len(plot_include) + 1 + len(baselines['PACS'])
nrows = sum(len(v) for v in datasets.values())
fig, axes = plt.subplots(
    nrows=nrows, ncols=ncols, figsize=(3*ncols, 3*nrows),
    constrained_layout=False, gridspec_kw=dict(wspace=0., hspace=0.15)
)
# share x within each column
# for c in range(ncols):
#     for r in range(1, nrows):
#         axes[r, c].sharex(axes[0, c])

# share y within each row
for r in range(nrows):
    # axes[r, 0].set_ylim(-0.05, 1.05)
    for c in range(1, ncols):
        axes[r, c].sharey(axes[r, 0])

# axes[0, 0].set_ylim(0.72, 1.0)
# for c in range(1, ncols):
#     axes[0, c].sharey(axes[0, 0])
# axes[1, 0].set_ylim(0.35, 1.03)
# for c in range(1, ncols):
#     axes[1, c].sharey(axes[1, 0])

# axes[0, 0].set_ylim(0.01, 0.18)
# for c in range(1, ncols):
#     axes[0, c].sharey(axes[0, 0])
# axes[1, 0].set_ylim(0.02, 0.16)
# for c in range(1, ncols):
#     axes[1, c].sharey(axes[1, 0])
# axes[2, 0].set_ylim(0.01, 0.37)
# for c in range(1, ncols):
#     axes[2, c].sharey(axes[2, 0])

# keep track of per-column min/max while you plot
col_minmax = [[np.inf, -np.inf] for _ in range(ncols)]
col_idx = 0

avg_metrics = {}

for task, dists in datasets.items():
    for dist in dists:
        file_path = f"metrics/{task}_model_{dist}{circuit_type}_data.csv"
        ID_name, OOD_name = get_ID_OOD_name(task, dist)
        if not os.path.exists(file_path):
            col_idx += 1
            continue
        df = pd.read_csv(file_path, comment="#")
        df = df.drop(columns=["model_id", "linear_probe", "weight_decay", "model_type", "use_adam", "learning_rate"], errors='ignore')
        # df["spectral_gap"] = np.abs(df["spectral_gap"])

        target_col = task_to_target[task](dist)
        df[target_col] = transform(df[target_col], y_transform)

        if dist != 'id':
            df['val_id_acc'] = transform(df['val_id_acc'], y_transform)

        # Log transform
        for col in log_include:
            if col in df.columns and np.issubdtype(df[col].dtype, np.number):
                df[col] = np.log(df[col].clip(lower=eps) + 3)
                if col == 'shortcut_vs_deep_ratio':
                    df[col] = np.log(df[col].clip(lower=eps)-0.7)
                elif col == 'edge_start_ratio_deep_vs_shallow_1':
                    df[col] = np.log(df[col].clip(lower=eps))
                # if dist == 'location_43' or dist == 'location_46':
                #     df[col] = np.log(df[col].clip(lower=eps) + 6)
                # elif task == 'metashift':
                #     df[col] = np.log(df[col].clip(lower=eps) + 4)
                # elif dist == 'hospital2':
                #     df[col] = np.log(df[col].clip(lower=eps))
                # elif 'PACS-photo' in task and 'cartoon' in dist:
                #     df[col] = np.log(df[col].clip(lower=eps) + 100)
                # elif 'PACS' in task:
                #     df[col] = np.log(df[col].clip(lower=eps) + 3)
                # else:
                #     df[col] = np.log1p(df[col].clip(lower=eps))

        # Plot target vs each metric
        metrics_to_plot = [col for col in df.columns if 'acc' not in col]
        for row_idx, metric in zip([0, -1, -2, -3, -4, -5, -6, -7, -8,-9,-10,-11,-12], ['val_id_acc'] + plot_include):
            if metric not in df.columns or target_col not in df.columns:
                continue

            ax = axes[col_idx][row_idx]
            x = df[metric].fillna(0).values.reshape(-1, 1)
            x1d = np.asarray(x).ravel()
            if x1d.size:
                col_minmax[row_idx][0] = min(col_minmax[row_idx][0], np.nanmin(x1d))
                col_minmax[row_idx][1] = max(col_minmax[row_idx][1], np.nanmax(x1d))
            y = df[target_col].values

            if len(x) == 0 or len(y) == 0:
                continue

            reg = LinearRegression().fit(x, y)
            y_pred = reg.predict(x)
            r2 = r2_score(y, y_pred)
            rho = safe_spearman(x[:, 0], y)
            tau = safe_kendall(x[:, 0], y)

            if metric in avg_metrics.keys():
                avg_metrics[metric].append(torch.tensor([r2, rho, tau]))
            else:
                avg_metrics[metric] = [torch.tensor([r2, rho, tau])]

            ax.scatter(x, y, alpha=0.6)
            ax.plot(x, y_pred, color='red')
            # if row_idx == 0:
                # ax.set_title(task_name_map[f"{task}-{dist}"])
                # ax.set_title(f"{task}-{dist}")
            ax.text(0.05, 0.95, f"$R^2$={r2:.3f}\nSRCC={rho:.3f}\nKRCC={tau:.3f}", transform=ax.transAxes, va='top', ha='left', fontsize=10,
                    bbox=dict(boxstyle="round,pad=0.3", edgecolor='gray', facecolor='white', alpha=0.8))
            # if 'terra' in task:
            #     ax.set_ylabel('OOD F1', fontsize=11)
            # else:
            #     ax.set_ylabel('OOD Accuracy', fontsize=11)
            if row_idx == 0:
            #     ax.text(
            #         -0.27 if 'PACS' in task else -0.3, 0.5,  # adjust -0.15 if too far/close
            #         task_2_title[task],
            #         fontweight="bold",
            #         fontsize=13,
            #         rotation=90,
            #         va="center", ha="center",
            #         transform=ax.transAxes
            #     )
                ax.set_ylabel(fr"${ID_name}\to {OOD_name}$")
            if col_idx == nrows - 1:
                ax.set_xlabel(metric_name_map[metric] if metric in metric_name_map.keys() else metric, fontsize=11)
            # ax.set_ylabel("OOD Accuracy")

        col_idx += 1

from sklearn.model_selection import KFold
col_idx = 0
# Add an extra row for multivariate fit
# for task, dists in datasets.items():
#     for dist in dists:
#         file_path = f"metrics/{task}_model_{dist}{circuit_type}_data.csv"
#         if not os.path.exists(file_path):
#             continue
#         df = pd.read_csv(file_path, comment="#")
#         df = df.drop(columns=["model_id", "linear_probe", "weight_decay", "model_type", "use_adam", "learning_rate"], errors='ignore')
#         df["spectral_gap"] = np.abs(df["spectral_gap"])
#
#         target_col = task_to_target[task](dist)
#         df[target_col] = transform(df[target_col], y_transform)
#
#         if dist != 'id':
#             df['val_id_acc'] = transform(df['val_id_acc'], y_transform)
#
#         for col in log_include:
#             if col in df.columns and np.issubdtype(df[col].dtype, np.number):
#                 if col == 'path_redundancy':
#                     df[col] += 1
#                 df[col] = np.log(df[col].clip(lower=eps))
#
#         X = df[plot_include].fillna(0).values
#         y = df[target_col].values
#
#         ax = axes[len(plot_include)+1][col_idx]  # last row
#
#         kf = KFold(n_splits=3, shuffle=True, random_state=42)
#         colors = ['red', 'green', 'blue']
#         for i, (train_idx, test_idx) in enumerate(kf.split(X)):
#             X_train, X_test = X[train_idx], X[test_idx]
#             y_train, y_test = y[train_idx], y[test_idx]
#
#             reg = LinearRegression().fit(X_train, y_train)
#             y_pred = reg.predict(X_test)
#             r2 = r2_score(y_test, y_pred)
#
#             ax.scatter(y_test, y_pred, alpha=0.6, label=f"Fold {i+1} ($R^2$={r2:.2f})", color=colors[i])
#
#         ax.plot([y.min(), y.max()], [y.min(), y.max()], color='black', linestyle='--')
#         ax.set_xlabel("True")
#         ax.set_ylabel("Predicted")
#         ax.set_title(f"{task}-{dist}\nMultivariate Fit")
#         ax.legend(fontsize=8)
#
#         col_idx += 1

# Add an extra row for multivariate fit with val_id_acc included
# col_idx = 0
# for task, dists in datasets.items():
#     for dist in dists:
#         file_path = f"metrics/{task}_model_{dist}{circuit_type}_data.csv"
#         if not os.path.exists(file_path):
#             col_idx += 1
#             continue
#         df = pd.read_csv(file_path, comment="#")
#         df = df.drop(columns=["model_id", "linear_probe", "weight_decay", "model_type", "use_adam", "learning_rate"], errors='ignore')
#         df["spectral_gap"] = np.abs(df["spectral_gap"])
#
#         target_col = task_to_target[task](dist)
#         df[target_col] = transform(df[target_col], y_transform)
#
#         if dist != 'id':
#             df['val_id_acc'] = transform(df['val_id_acc'], y_transform)
#
#         for col in log_include:
#             if col in df.columns and np.issubdtype(df[col].dtype, np.number):
#                 if col == 'path_redundancy':
#                     df[col] += 1
#                 df[col] = np.log(df[col].clip(lower=eps))
#
#         # Combine features: val_id_acc + plot_include
#         X_cols = ['val_id_acc'] + plot_include
#         X_cols = [c for c in X_cols if c in df.columns]
#         X = df[X_cols].fillna(0).values
#         y = df[target_col].values
#
#         ax = axes[len(plot_include)+1][col_idx]  # last row
#
#         kf = KFold(n_splits=3, shuffle=True, random_state=42)
#         colors = ['red', 'green', 'blue']
#         for i, (train_idx, test_idx) in enumerate(kf.split(X)):
#             X_train, X_test = X[train_idx], X[test_idx]
#             y_train, y_test = y[train_idx], y[test_idx]
#
#             reg = LinearRegression().fit(X_train, y_train)
#             y_pred = reg.predict(X_test)
#             r2 = r2_score(y_test, y_pred)
#
#             ax.scatter(y_test, y_pred, alpha=0.6, label=f"Fold {i+1} ($R^2$={r2:.2f})", color=colors[i])
#
#         ax.plot([y.min(), y.max()], [y.min(), y.max()], color='black', linestyle='--')
#         ax.set_xlabel("True")
#         ax.set_ylabel("Predicted")
#         ax.set_title(f"{task}-{dist}\nMultivariate + val_id_acc")
#         ax.legend(fontsize=8)
#
#         col_idx += 1

# === Baseline rows (univariate) ===
baseline_start_row = 1  # rows after: [univariate], [multi], [multi+val_id_acc]
col_idx = 0

for task, dists in datasets.items():
    for dist in dists:
        file_path = f"metrics/{task}_model_{dist}{circuit_type}_data.csv"
        if not os.path.exists(file_path):
            col_idx += 1
            continue

        df = pd.read_csv(file_path, comment="#")
        df = df.drop(columns=["model_id", "linear_probe", "weight_decay", "model_type", "use_adam", "learning_rate"], errors='ignore')
        # df["spectral_gap"] = np.abs(df["spectral_gap"])

        target_col = task_to_target[task](dist)
        df[target_col] = transform(df[target_col], y_transform)

        if dist != 'id':
            df['val_id_acc'] = transform(df['val_id_acc'], y_transform)

        # same log transforms you applied above (keeps consistency)
        for col in log_include:
            if col in df.columns and np.issubdtype(df[col].dtype, np.number):
                if col == 'path_redundancy':
                    df[col] += 1
                df[col] = np.log(df[col].clip(lower=eps))

        # derive readable baseline labels from the lambda-generated column names
        def baseline_label(fn):
            name = fn("DIST")  # e.g., "test_AC_DIST"
            name = name.replace("test_", "").replace("_DIST", "")
            return name.upper()

        if task not in baselines:
            # hide the baseline axes for this column if no baselines defined
            for i in range(len(baselines["PACS"])):
                axes[col_idx][baseline_start_row + i].set_visible(False)
            col_idx += 1
            continue

        for i, fn in enumerate(baselines[task]):
            ax = axes[col_idx][baseline_start_row + i]
            bcol = fn(dist)  # e.g., "test_AC_photo"
            if 'EMD' in bcol:
                df[bcol] = np.log(df[bcol].clip(lower=eps))

            label = baseline_label(fn)

            if bcol not in df.columns:
                # no baseline value for this (task,dist); hide this subplot
                ax.set_visible(False)
                continue

            x = df[bcol].values.reshape(-1, 1)
            x1d = np.asarray(x).ravel()
            if x1d.size:
                col_minmax[baseline_start_row + i][0] = min(col_minmax[baseline_start_row + i][0], np.nanmin(x1d))
                col_minmax[baseline_start_row + i][1] = max(col_minmax[baseline_start_row + i][1], np.nanmax(x1d))
            y = df[target_col].values
            if len(x) == 0 or len(y) == 0:
                ax.set_visible(False)
                continue

            reg = LinearRegression().fit(x, y)
            y_pred = reg.predict(x)
            r2 = r2_score(y, y_pred)
            rho = safe_spearman(x[:, 0], y)
            tau = safe_kendall(x[:, 0], y)
            if label in avg_metrics.keys():
                avg_metrics[label].append(torch.tensor([r2, rho, tau]))
            else:
                avg_metrics[label] = [torch.tensor([r2, rho, tau])]

            # plot points + fitted line (sorted for a clean line)
            order = np.argsort(x[:, 0])
            ax.scatter(x, y, alpha=0.6)
            ax.plot(x[order], y_pred[order], color='red')

            ax.text(0.05, 0.95, f"$R^2$={r2:.3f}\nSRCC={rho:.3f}\nKRCC={tau:.3f}", transform=ax.transAxes, va='top', ha='left', fontsize=10,
                    bbox=dict(boxstyle="round,pad=0.3", edgecolor='gray', facecolor='white', alpha=0.8))
            if col_idx == nrows - 1:
                ax.set_xlabel(label if label != 'EMD' else 'MDE')
#             ax.set_ylabel("OOD Accuracy")
#             if 'terra' in task:
#                 ax.set_ylabel('OOD F1', fontsize=11)
#             else:
#                 ax.set_ylabel('OOD Accuracy', fontsize=11)

        col_idx += 1
# row_titles = [
#     r'PACS (photo$\to$sketch)',
#     r'Camelyon17 (ID$\to$Hospital1)',
#     r'Terra Incognita (L38$\to$L46)',
# ]
# apply consistent limits within each column + tidy labels
def expand_range(xmin, xmax, frac=0.05):
    span = xmax - xmin
    return xmin - span*frac, xmax + span*frac
# for c in range(ncols):
#     xmin, xmax = col_minmax[c]
#     if not np.isfinite([xmin, xmax]).all():
#         continue
#     xmin, xmax = expand_range(xmin,xmax)
#     for r in range(nrows):
#         axes[r, c].set_xlim(xmin, xmax)

# only show x tick labels on the bottom row
# for r in range(nrows-1):
#     for c in range(ncols):
#         axes[r, c].tick_params(labelbottom=False)

# keep y labels only on the leftmost column for compactness
for r in range(nrows):
    for c in range(1, ncols):
        axes[r, c].tick_params(labelleft=False)

# for ax in axes.flat:
#     ax.margins(x=0.03, y=0.05)

for name, value in avg_metrics.items():
    val = torch.stack(value)
    avg_val = val.mean(dim=0)
    print(f'metric {name}: {avg_val}')

# place a centered title above each row
# plt.tight_layout(rect=(0, 0, 1, 0.97))
# fig.subplots_adjust(hspace=0.51, top=0.94)
# for axrow, title in zip(axes, row_titles):
#     y = axrow[0].get_position().y1 + 0.02   # a little above the row
#     fig.text(0.5, y, title, ha='center', va='bottom',
#              fontsize=14, fontweight='bold')
# plt.savefig("figures/fit_plots_PACS.pdf",
#             bbox_inches="tight",   # trim empty margins
#             pad_inches=0.01)       # tiny extra pad to avoid clipping

plt.savefig("figures/fit_plots_camelyon17.pdf", transparent=True)       # tiny extra pad to avoid clipping

