import os

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from scipy.stats import norm, boxcox, yeojohnson
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

datasets = {
    'PACS': ['photo', 'cartoon', 'art_painting', 'id'],
    'camelyon17': ['id', 'hospital1', 'hospital2'],
    'metashift': ['id', 'ood'],
    'terra-incognita-38': ['id', 'location43', 'location46', 'location100']
}
task_to_target = {
    'PACS': lambda d: 'val_id_acc' if d == 'id' else f'test_acc_{d}',
    'camelyon17': lambda d: 'val_id_acc' if d == 'id' else f'test_acc_{d}',
    'metashift': lambda d: 'val_id_acc' if d == 'id' else f'test_id_acc',
    'terra-incognita-38': lambda d: 'val_id_acc' if d == 'id' else f'test_acc_{d}'
}
circuit_type = '_kl'
y_transform = 'probit'
plot_include = [
    'logit_contribution_ratio_deep_vs_shallow',
    'circuit_instability',
    'modularity',
    'avg_path_length',
    'effective_path_depth',
    'weighted_shortcut_score',
    'avg_edge_flow_centrality',
    'spectral_radius',
    'normed_spectral_radius',
    'weighted_path_depth',
    'layerwise_score_entropy',
    'path_depth_entropy',
    'laplacian_energy',
    'normed_laplacian_energy',
    'algebraic_connectivity',
    'normed_algebraic_connectivity',
    'normed_graph_energy',
    'edge_score_gini',
    'avg_betweenness',
    'path_redundancy',
    'avg_clustering',
    'laplacian_spectral_entropy',
    'std_entropy_across_layers',
    'spectral_gap',
    'community_size_variance',
    'layerwise_score_entropy',
    'attention_mlp_ratio',
]
log_include = {
    'circuit_instability',
    'logit_contribution_ratio_deep_vs_shallow',
    'spectral_gap',
    'avg_path_length',
    'effective_path_depth',
    'spectral_radius',
    'weighted_path_depth',
    'normed_graph_energy',
    'avg_edge_flow_centrality',

    # 'layerwise_score_entropy'
}

# --------- Plotting Setup ---------
fig, axes = plt.subplots(
    nrows=len(plot_include) + 3,
    ncols=sum(len(v) for v in datasets.values()),
    figsize=(4 * sum(len(v) for v in datasets.values()), 3 * (len(plot_include) + 3)),
    squeeze=False
)
col_idx = 0

for task, dists in datasets.items():
    for dist in dists:
        file_path = f"metrics/{task}_model_id{circuit_type}_data.csv"
        if not os.path.exists(file_path):
            col_idx += 1
            continue
        df = pd.read_csv(file_path, comment="#")
        df = df.drop(columns=["model_id", "linear_probe", "weight_decay", "model_type", "use_adam", "learning_rate"], errors='ignore')
        df["spectral_gap"] = np.abs(df["spectral_gap"])

        target_col = task_to_target[task](dist)
        df[target_col] = transform(df[target_col], y_transform)

        if dist != 'id':
            df['val_id_acc'] = transform(df['val_id_acc'], y_transform)

        # Log transform
        for col in log_include:
            if col in df.columns and np.issubdtype(df[col].dtype, np.number):
                if col == 'path_redundancy':
                    df[col] += 1
                df[col] = np.log(df[col].clip(lower=eps))

        # Plot target vs each metric
        metrics_to_plot = [col for col in df.columns if 'acc' not in col]
        for row_idx, metric in enumerate(['val_id_acc'] + plot_include):
            if metric not in df.columns or target_col not in df.columns:
                continue

            ax = axes[row_idx][col_idx]
            x = df[metric].fillna(0).values.reshape(-1, 1)
            y = df[target_col].values

            if len(x) == 0 or len(y) == 0:
                continue

            reg = LinearRegression().fit(x, y)
            y_pred = reg.predict(x)
            r2 = r2_score(y, y_pred)

            ax.scatter(x, y, alpha=0.6)
            ax.plot(x, y_pred, color='red')
            ax.set_title(f"{task}-{dist}")
            ax.text(0.05, 0.95, f"$R^2$={r2:.2f}", transform=ax.transAxes, va='top', ha='left', fontsize=10,
                    bbox=dict(boxstyle="round,pad=0.3", edgecolor='gray', facecolor='white', alpha=0.8))
            ax.set_xlabel(metric)
            ax.set_ylabel(target_col)

        col_idx += 1

from sklearn.model_selection import KFold
col_idx = 0
# Add an extra row for multivariate fit
for task, dists in datasets.items():
    for dist in dists:
        file_path = f"metrics/{task}_model_{dist}{circuit_type}_data.csv"
        if not os.path.exists(file_path):
            continue
        df = pd.read_csv(file_path, comment="#")
        df = df.drop(columns=["model_id", "linear_probe", "weight_decay", "model_type", "use_adam", "learning_rate"], errors='ignore')
        df["spectral_gap"] = np.abs(df["spectral_gap"])

        target_col = task_to_target[task](dist)
        df[target_col] = transform(df[target_col], y_transform)

        if dist != 'id':
            df['val_id_acc'] = transform(df['val_id_acc'], y_transform)

        for col in log_include:
            if col in df.columns and np.issubdtype(df[col].dtype, np.number):
                if col == 'path_redundancy':
                    df[col] += 1
                df[col] = np.log(df[col].clip(lower=eps))

        X = df[plot_include].fillna(0).values
        y = df[target_col].values

        ax = axes[len(plot_include)+1][col_idx]  # last row

        kf = KFold(n_splits=3, shuffle=True, random_state=42)
        colors = ['red', 'green', 'blue']
        for i, (train_idx, test_idx) in enumerate(kf.split(X)):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            reg = LinearRegression().fit(X_train, y_train)
            y_pred = reg.predict(X_test)
            r2 = r2_score(y_test, y_pred)

            ax.scatter(y_test, y_pred, alpha=0.6, label=f"Fold {i+1} ($R^2$={r2:.2f})", color=colors[i])

        ax.plot([y.min(), y.max()], [y.min(), y.max()], color='black', linestyle='--')
        ax.set_xlabel("True")
        ax.set_ylabel("Predicted")
        ax.set_title(f"{task}-{dist}\nMultivariate Fit")
        ax.legend(fontsize=8)

        col_idx += 1

# Add an extra row for multivariate fit with val_id_acc included
col_idx = 0
for task, dists in datasets.items():
    for dist in dists:
        file_path = f"metrics/{task}_model_{dist}{circuit_type}_data.csv"
        if not os.path.exists(file_path):
            col_idx += 1
            continue
        df = pd.read_csv(file_path, comment="#")
        df = df.drop(columns=["model_id", "linear_probe", "weight_decay", "model_type", "use_adam", "learning_rate"], errors='ignore')
        df["spectral_gap"] = np.abs(df["spectral_gap"])

        target_col = task_to_target[task](dist)
        df[target_col] = transform(df[target_col], y_transform)

        if dist != 'id':
            df['val_id_acc'] = transform(df['val_id_acc'], y_transform)

        for col in log_include:
            if col in df.columns and np.issubdtype(df[col].dtype, np.number):
                if col == 'path_redundancy':
                    df[col] += 1
                df[col] = np.log(df[col].clip(lower=eps))

        # Combine features: val_id_acc + plot_include
        X_cols = ['val_id_acc'] + plot_include
        X_cols = [c for c in X_cols if c in df.columns]
        X = df[X_cols].fillna(0).values
        y = df[target_col].values

        ax = axes[len(plot_include)+2][col_idx]  # last row

        kf = KFold(n_splits=3, shuffle=True, random_state=42)
        colors = ['red', 'green', 'blue']
        for i, (train_idx, test_idx) in enumerate(kf.split(X)):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            reg = LinearRegression().fit(X_train, y_train)
            y_pred = reg.predict(X_test)
            r2 = r2_score(y_test, y_pred)

            ax.scatter(y_test, y_pred, alpha=0.6, label=f"Fold {i+1} ($R^2$={r2:.2f})", color=colors[i])

        ax.plot([y.min(), y.max()], [y.min(), y.max()], color='black', linestyle='--')
        ax.set_xlabel("True")
        ax.set_ylabel("Predicted")
        ax.set_title(f"{task}-{dist}\nMultivariate + val_id_acc")
        ax.legend(fontsize=8)

        col_idx += 1

plt.tight_layout()
plt.savefig('figures/fit_plots_intrinsic.png')
