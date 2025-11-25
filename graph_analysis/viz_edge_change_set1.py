import os.path

import pandas as pd
import torch
from scipy.stats import pearsonr, spearmanr
from sklearn.cross_decomposition import PLSRegression, CCA
from sklearn.model_selection import KFold
from sklearn.decomposition import NMF
from torch.optim import Adam
from tqdm import tqdm
from sklearn.metrics.pairwise import rbf_kernel, linear_kernel
import pickle

from eap.graph import Graph

import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg as la
import types

# only patch if needed
if "eigvals" not in la.eigh.__code__.co_varnames:
    _orig_eigh = la.eigh

    def _patched_eigh(a, b=None, eigvals=None, **kwargs):
        # map deprecated eigvals=(lo, hi) to subset_by_index=(lo, hi)
        if eigvals is not None:
            lo, hi = eigvals
            kwargs["subset_by_index"] = (lo, hi)
        return _orig_eigh(a, b, **kwargs)

    la.eigh = types.FunctionType(_patched_eigh.__code__, globals(), "eigh")
from rcca import CCA as RCCA

def build_rank_delta_matrix(edges_ood, node_list, granularity="layer"):
    """
    Build adjacency matrix where entry [i, j] = rank_delta or score of edge i->j.

    Args:
        edges_ood (dict): { (src->dst<qkv>): importance score }
        node_list (list): ordered list of node names (for node-level granularity)
        granularity (str): "layer" (default) or "node"

    Returns:
        np.ndarray: adjacency matrix (layer x layer OR node x node)
    """

    all_edges = list(edges_ood.keys())
    ood_scores = np.array([edges_ood[e].abs().item() for e in all_edges])

    # --- Helper: map node to layer index ---
    def get_layer(n):
        if n.startswith("a"):
            return int(n.split(".")[0][1:]) + 1   # attn head -> layer id
        elif n.startswith("m"):
            return int(n[1:]) + 1                 # mlp -> layer id
        elif n == "logits":
            return 12 + 1
        else:
            return 0

    if granularity == "layer":
        n_layers = 14
        mat = np.zeros((n_layers, n_layers))
        counts = np.zeros((n_layers, n_layers))

        for edge, score in zip(all_edges, ood_scores):
            src, dst = edge.split("->")[0], edge.split("->")[1].split("<")[0]
            i, j = get_layer(src), get_layer(dst)
            mat[i, j] += score
            counts[i, j] += 1

        # mat /= counts + 1e-12
        return mat

    elif granularity == "node":
        node_to_idx = {n: i for i, n in enumerate(node_list)}
        N = len(node_list)
        mat = np.zeros((N, N))
        counts = np.zeros((N, N))

        for edge, score in zip(all_edges, ood_scores):
            src, dst = edge.split("->")[0], edge.split("->")[1].split("<")[0]
            if src not in node_to_idx or dst not in node_to_idx:
                continue
            i, j = node_to_idx[src], node_to_idx[dst]
            mat[i, j] += score
            counts[i, j] += 1

        mat /= counts + 1e-12
        return mat

    else:
        raise ValueError("granularity must be 'layer' or 'node'")

def analyze_circuits_with_cca(mats, test_accs, n_components=1, plot=True, rank_corr=True, task='', domain=''):
    """
    Use Canonical Correlation Analysis (CCA) to extract components maximally
    correlated with OOD accuracy, and visualize them as adjacency matrices
    and scatter plots.

    Parameters
    ----------
    mats : dict[int -> np.ndarray]
        Mapping from model_id to adjacency matrix (L x L).
    test_accs : dict[int -> float]
        Mapping from model_id to OOD accuracy.
    n_components : int
        Number of canonical components to extract.
    plot : bool
        Whether to plot component heatmaps and scatter plots.
    rank_corr : bool
        If True, report Spearman correlation; else Pearson.

    Returns
    -------
    W : np.ndarray
        Canonical component scores per model (n_models x n_components).
    comp_matrices : np.ndarray
        Component loadings reshaped into adjacency matrices (n_components x L x L).
    correlations : list[dict]
        Pearson/Spearman correlations for each component.
    """
    model_ids = sorted(mats.keys())
    accs = np.array([test_accs[mid] for mid in model_ids])
    matrices = np.stack([mats[mid] for mid in model_ids], axis=0)

    n_models, L, _ = matrices.shape
    X = matrices.reshape(n_models, L * L)
    y = accs.reshape(-1, 1)

    cca = CCA(n_components=n_components)
    X_c, y_c = cca.fit_transform(X, y)

    # Flip if correlation is negative
    corr = np.corrcoef(X_c[:, 0], y.ravel())[0, 1]
    if corr < 0:
        X_c *= -1
        cca.x_loadings_ *= -1

    # Extract loadings → reshape to adjacency matrices
    comp_matrices = cca.x_loadings_.T.reshape(n_components, L, L)

    # Correlations
    correlations = []
    for i in range(n_components):
        if rank_corr:
            rho, _ = spearmanr(X_c[:, i], accs)
            correlations.append({"component": i, "spearman_rho": rho})
        else:
            r, _ = pearsonr(X_c[:, i], accs)
            correlations.append({"component": i, "pearson_r": r})

    if plot:
        fig, axes = plt.subplots(2, n_components, figsize=(4 * n_components, 8))

        # Top row: component heatmap
        im = axes[0].imshow(comp_matrices[0], cmap="viridis")
        axes[0].set_title(f"CCA Comp {0}")
        axes[0].invert_yaxis()
        plt.colorbar(im, ax=axes[0], fraction=0.046, pad=0.04)

        # Bottom row: scatter plot of coeff vs accuracy
        axes[1].scatter(X_c[:, 0], accs, alpha=0.7)
        axes[1].set_xlabel(f"Comp {0} score")
        axes[1].set_ylabel("OOD Acc")

        if rank_corr:
            axes[1].set_title(f"rho={correlations[0]['spearman_rho']:.2f}")
        else:
            axes[1].set_title(f"r={correlations[0]['pearson_r']:.2f}")

        plt.tight_layout()
        plt.savefig(f'./figures/CCAs_pdf_reverse/{task}-{domain}_viz_node.pdf')

    return X_c, comp_matrices, correlations

def analyze_circuits_with_cca_cv(
    mats, test_accs, n_components=1, reg_param=1e-3, kcca=False, kernel="rbf",
    n_splits=5, plot=True, rank_corr=True, task='', domain=''
):
    """
    Regularized and cross-validated Canonical Correlation Analysis (CCA)
    between circuit adjacency matrices and OOD accuracies.
    Supports both linear and kernel CCA.

    Parameters
    ----------
    mats : dict[int -> np.ndarray]
        Mapping from model_id to adjacency matrix (L x L).
    test_accs : dict[int -> float]
        Mapping from model_id to OOD accuracy.
    n_components : int
        Number of canonical components to extract.
    reg_param : float
        Regularization parameter for RCCA.
    kcca : bool
        Whether to use kernel CCA.
    kernel : str
        Kernel type for KCCA ('rbf' or 'linear').
    n_splits : int
        Number of folds for KFold cross-validation.
    plot : bool
        Whether to visualize the canonical component and scatter.
    rank_corr : bool
        If True, use Spearman correlation; else Pearson.
    task, domain : str
        Labels for saving visualizations.

    Returns
    -------
    results : dict
        mean_corr, fold_corrs, canonical_vec, component_matrix
    """
    model_ids = sorted(mats.keys())
    accs = np.array([test_accs[mid] for mid in model_ids])
    matrices = np.stack([mats[mid] for mid in model_ids], axis=0)

    n_models, L, _ = matrices.shape
    X = matrices.reshape(n_models, L * L)
    y = accs.reshape(-1, 1)

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    fold_corrs = []

    # Define kernel function
    def compute_kernel(A, B):
        if kernel == "rbf":
            return rbf_kernel(A, B, gamma=None)
        elif kernel == "linear":
            return linear_kernel(A, B)
        else:
            raise ValueError(f"Unsupported kernel type: {kernel}")

    def center_kernel(K):
        """Double-center the kernel matrix."""
        n = K.shape[0]
        H = np.eye(n) - np.ones((n, n)) / n
        return H @ K @ H

    # --- Cross-validation ---
    for fold, (train_idx, test_idx) in enumerate(kf.split(X)):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        if kcca:
            gamma = 1.0 / (2 * np.median(np.linalg.norm(X_train[:, None] - X_train, axis=2)) ** 2)
            X_train_k = rbf_kernel(X_train, X_train, gamma=gamma)
            y_train_k = rbf_kernel(y_train, y_train, gamma=1.0)
            X_train_k = center_kernel(X_train_k)
            y_train_k = center_kernel(y_train_k)
            gamma = 1.0 / (2 * np.median(np.linalg.norm(X_test[:, None] - X_test, axis=2)) ** 2)
            X_test_k = rbf_kernel(X_test, X_train, gamma=gamma)
            y_test_k = rbf_kernel(y_test, y_train, gamma=1.0)
            X_test_k = center_kernel(X_test_k)
            y_test_k = center_kernel(y_test_k)
            rcca = RCCA(kernelcca=True, numCC=1, reg=reg_param)
            rcca.train([X_train_k, y_train_k])
            X_c_test = X_test_k @ rcca.comps[0]
            y_c_test = y_test_k @ rcca.comps[1]
        else:
            rcca = RCCA(kernelcca=False, numCC=1, reg=reg_param)
            rcca.train([X_train, y_train])
            X_c_test = X_test @ rcca.ws[0]
            y_c_test = y_test @ rcca.ws[1]

        rho, _ = spearmanr(X_c_test[:, 0], y_c_test.ravel()) if rank_corr else pearsonr(X_c_test[:, 0], y_c_test.ravel())
        fold_corrs.append(rho)

    mean_corr = np.mean(fold_corrs)

    # --- Fit full data for visualization ---
    if kcca:
        gamma = 1.0 / (2 * np.median(np.linalg.norm(X[:, None] - X, axis=2)) ** 2)
        X_k = rbf_kernel(X, X, gamma=gamma)
        y_k = rbf_kernel(y, y, gamma=1.0)

        # Center kernels
        X_k = center_kernel(X_k)
        y_k = center_kernel(y_k)
        rcca = RCCA(kernelcca=True, numCC=1, reg=0.01)
        rcca.train([X_k, y_k])
        X_c_full = X_k @ rcca.comps[0]
        y_c_full = y_k @ rcca.comps[1]
        # feature-level correlation as proxy for importance
        alpha = rcca.comps[0]  # dual weights
        X_c = X_k @ alpha[:, 0]  # canonical projection

        # Approximate feature importance by correlation between each feature and canonical projection
        canonical_vec = np.array([
            np.corrcoef(X[:, k], X_c)[0, 1] for k in range(X.shape[1])
        ])
    else:
        rcca = RCCA(kernelcca=False, numCC=1, reg=reg_param)
        rcca.train([X, y])
        X_c_full = X @ rcca.ws[0]
        y_c_full = y @ rcca.ws[1]
        canonical_vec = rcca.ws[0][:, 0]

    corr = np.corrcoef(X_c_full[:, 0], y.ravel())[0, 1]
    if corr < 0:
        X_c_full *= -1
        canonical_vec *= -1

    component_matrix = canonical_vec.reshape(L, L)

    # --- Plot ---
    if plot:
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        im = axes[0].imshow(component_matrix, cmap="viridis")
        axes[0].set_title(f"{'KCCA' if kcca else 'CCA'} Comp (reg={reg_param})")
        plt.colorbar(im, ax=axes[0], fraction=0.046, pad=0.04)
        axes[1].scatter(X_c_full[:, 0], accs, alpha=0.7)
        axes[1].set_xlabel("Canonical Component Score")
        axes[1].set_ylabel("OOD Accuracy")
        axes[1].set_title(f"Mean CV corr={mean_corr:.2f}")
        plt.tight_layout()
        plt.savefig(f'./figures/CCAs_reg/{task}-{domain}_cca_reg{reg_param}_k{kcca}.png')

    return {
        'mean_corr': mean_corr,
        'fold_corrs': fold_corrs,
        'canonical_vec': canonical_vec,
        'component_matrix': component_matrix
    }

def analyze_circuits_with_pls(mats, test_accs, n_components=5, plot=True):
    """
    Use PLS regression to extract components aligned with OOD accuracy,
    and visualize them as adjacency matrices and scatter plots.
    """
    model_ids = sorted(mats.keys())
    accs = np.array([test_accs[mid] for mid in model_ids])
    matrices = np.stack([mats[mid] for mid in model_ids], axis=0)

    n_models, L, _ = matrices.shape
    X = matrices.reshape(n_models, L * L)

    pls = PLSRegression(n_components=n_components, scale=True)
    W, Y_pred = pls.fit_transform(X, accs)

    # Extract loadings for each component → reshape to adjacency matrices
    comp_matrices = pls.x_weights_.T.reshape(n_components, L, L)

    # Correlations
    correlations = []
    for i in range(n_components):
        pear, _ = pearsonr(W[:, i], accs)
        spear, _ = spearmanr(W[:, i], accs)
        correlations.append({"component": i, "pearson_r": pear, "spearman_rho": spear})

    if plot:
        fig, axes = plt.subplots(2, n_components, figsize=(4 * n_components, 8))

        for i in range(n_components):
            # Top row: component heatmap
            im = axes[0, i].imshow(comp_matrices[i], cmap="viridis")
            axes[0, i].set_title(f"PLS Comp {i}")
            plt.colorbar(im, ax=axes[0, i], fraction=0.046, pad=0.04)

            # Bottom row: scatter plot of coeff vs accuracy
            axes[1, i].scatter(W[:, i], accs, alpha=0.7)
            axes[1, i].set_xlabel(f"Comp {i} score")
            axes[1, i].set_ylabel("OOD Acc")
            axes[1, i].set_title(
                f"r={correlations[i]['pearson_r']:.2f}, rho={correlations[i]['spearman_rho']:.2f}"
            )

        plt.tight_layout()
        plt.savefig('./figures/PLS_viz_node.png')

    return W, comp_matrices, correlations

def plot_rank_delta_matrices(mat_dict, test_accs, node_list,
                             title_prefix="Rank Delta Matrix",
                             cmap="coolwarm",
                             vmin=0, vmax=9000):
    """
    Plot rank-delta matrices stored in a dict, sorted by test accuracy.

    Args:
        mat_dict: dict[str, np.ndarray]  # domain -> matrix
        test_accs: dict[str, float]      # domain -> test accuracy
        node_list: list[str]             # labels for x/y ticks
        outdir: str, directory to save figures
        title_prefix: str
        cmap: str, matplotlib colormap
        vmin, vmax: float, colorbar range
    """
    # ---- 1. sort domains by test accuracy (descending)
    domains_sorted = sorted(test_accs.keys(),
                            key=lambda k: test_accs[k],
                            reverse=True)

    n_domains = len(domains_sorted)

    # 2. choose subplot grid
    ncols = min(3, n_domains)  # up to 3 per row
    nrows = int(np.ceil(n_domains / ncols))

    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(5 * ncols, 4 * nrows),
                             squeeze=False)

    # 3. plot each domain
    ims = []
    for idx, domain in enumerate(domains_sorted):
        r, c = divmod(idx, ncols)
        ax = axes[r, c]
        acc = test_accs[domain]
        mat = mat_dict[domain]

        vmax = mat.max()

        im = ax.imshow(mat, cmap=cmap, interpolation="nearest",
                       vmin=vmin, vmax=vmax)
        ims.append(im)

        ax.set_xticks(range(len(node_list)))
        ax.set_xticklabels(node_list, rotation=90)
        ax.set_yticks(range(len(node_list)))
        ax.set_yticklabels(node_list)
        ax.set_title(f"{title_prefix} — {domain} (Acc={acc:.3f})")

    # remove any unused axes
    for idx in range(len(domains_sorted), nrows * ncols):
        r, c = divmod(idx, ncols)
        fig.delaxes(axes[r, c])

    # 4. shared colorbar
    cbar = fig.colorbar(ims[0], ax=axes, orientation="vertical",
                        fraction=0.02, pad=0.04)
    cbar.set_label("Δ Rank (OOD - ID)")

    plt.tight_layout()
    plt.savefig('./figures/edge_change_heatmap_camelyon17-set1.png')


def get_block_from_node_name(name):
    if name.startswith('a'):
        return int(name.split('.')[0][1:])  # e.g., a3.1 → 3
    elif name.startswith('m'):
        return int(name[1:])  # e.g., m6 → 6
    elif name == 'input':
        return -1
    elif name.startswith('logits'):
        return 99
    raise ValueError(f"Unknown node name: {name}")

def analyze_circuits_with_nmf(mats, test_accs, n_components=5, plot=True):
    """
    Perform NMF on adjacency matrices (circuit representations) and
    correlate discovered components with OOD accuracy.

    Args:
        mats (dict): {model_id: adjacency_matrix (numpy array, shape [L, L])}
        test_accs (dict): {model_id: float} OOD accuracies
        n_components (int): number of NMF components to extract
        plot (bool): whether to visualize basis matrices

    Returns:
        W (np.ndarray): coefficient matrix (n_models x n_components)
        H_matrices (np.ndarray): basis matrices (n_components x L x L)
        correlations (list): per-component correlation with OOD accuracy
    """
    model_ids = sorted(mats.keys())
    accs = np.array([test_accs[mid] for mid in model_ids])
    matrices = np.stack([mats[mid] for mid in model_ids], axis=0)  # shape (n_models, L, L)

    n_models, L, _ = matrices.shape
    X = matrices.reshape(n_models, L * L)  # flatten into (n_models, L^2)

    nmf = NMF(n_components=n_components, init="nndsvda", random_state=0, max_iter=2000)
    W = nmf.fit_transform(X)       # (n_models, n_components)
    H = nmf.components_            # (n_components, L^2)
    H_matrices = H.reshape(n_components, L, L)

    # Compute correlations
    correlations = []
    for i in range(n_components):
        pear, p_p = pearsonr(W[:, i], accs)
        spear, p_s = spearmanr(W[:, i], accs)
        correlations.append({
            "component": i,
            "pearson_r": pear, "pearson_p": p_p,
            "spearman_rho": spear, "spearman_p": p_s
        })

    if plot:
        fig, axes = plt.subplots(2, n_components, figsize=(4 * n_components, 8))

        for i in range(n_components):
            # Top row: basis adjacency matrix
            im = axes[0, i].imshow(H_matrices[i], cmap="viridis")
            axes[0, i].set_title(f"Comp {i}")
            plt.colorbar(im, ax=axes[0, i], fraction=0.046, pad=0.04)

            # Bottom row: scatter of coeffs vs acc
            axes[1, i].scatter(W[:, i], accs, alpha=0.7)
            axes[1, i].set_xlabel(f"Coeff Comp {i}")
            axes[1, i].set_ylabel("OOD Acc")

            # Annotate with correlations
            pear, spear = correlations[i]["pearson_r"], correlations[i]["spearman_rho"]
            axes[1, i].set_title(f"r={pear:.2f}, rho={spear:.2f}")

        plt.tight_layout()
        plt.savefig('./figures/NMF_viz_node.png')

    return W, H_matrices, correlations


model_id = 0
residual = False
include_id = True
TOPN = 500
tasks = {
    'PACS-art_painting': ['cartoon', 'photo', 'sketch'],
    'PACS': ['art_painting', 'cartoon', 'photo'],
    'PACS-photo' : ['art_painting', 'cartoon', 'sketch'],
    'PACS-cartoon': ['art_painting', 'photo', 'sketch'],
    'terra-incognita-38': ['location_43', 'location_46', 'location_100'],
    'terra-incognita-43': ['location_38', 'location_46', 'location_100'],
    'terra-incognita-46': ['location_43', 'location_38', 'location_100'],
    'terra-incognita-100': ['location_43', 'location_46', 'location_38'],
    'camelyon17': ['hospital1', 'hospital2']
}
if residual:
    resid_str = "_resid"
else:
    resid_str = ""
Hs = []
all_mats = []
all_accs = []

if os.path.exists('Hs_8.p'):
    with open('Hs_8.p', 'rb') as file:
        Hs, all_mats, all_accs = pickle.load(file)
else:
    for task, domains in tasks.items():
        for domain in domains:
            SWEEP_CSV = f"/home/yxpengcs/PycharmProjects/vit-spurious-robustness/output/{task}_sweep_results_new.csv"
            sweep = pd.read_csv(SWEEP_CSV)

            circuits = {}
            mats = {}
            test_accs = {}
            for row_id, row in tqdm(sweep.iterrows()):
                model_id = row.model_id
                if 'terra' in task:
                    test_accs[model_id] = row[f'test_f1_{domain}']
                else:
                    test_accs[model_id] = row[f'test_acc_{domain}']
                if 'location' in domain:
                    domain_name = domain.replace('_', '')
                else:
                    domain_name = domain

                gpath = (
                    f"/home/yxpengcs/PycharmProjects/vMIB-circuit/circuits/EAP-IG-inputs_mean-positional_edge_train_kl_divergence/"
                    f"{task.replace('_', '-')}-mean-{domain_name.replace('_', '-')}_sweep_{model_id}/importances.pt"
                )
                per_example_scores_path = gpath.replace("importances.pt", "perexample_importances.p")
                if residual:
                    gpath = (
                        f"/home/yxpengcs/PycharmProjects/vMIB-circuit/circuits/EAP-IG-inputs_mean-positional_edge_train_kl_divergence/"
                        f"{task.replace('_', '-')}-mean-{domain_name.replace('_', '-')}_sweep_{model_id}/residual_importances.pt"
                    )

                # if not (os.path.exists(gpath) and os.path.exists(per_example_scores_path)):
                #     print(f"[WARN] Missing graph or perexample scores for domain={domain}; skipping.")
                #     continue

                # with open(per_example_scores_path, "rb") as f:
                #     per_example_scores = pickle.load(f)
                edges = {}
                g = Graph.from_pt(gpath)
                for e in g.edges.values():
                    edges[f'{e.parent.name}->{e.child.name}<{e.qkv}>'] = e.score.abs()
                circuits[model_id] = edges
                # keep same pruning as your previous code
                if residual:
                    g.apply_topn(TOPN, True, level="edge", prune=True) # the negative residual also important
                else:
                    g.apply_topn(TOPN, False, level="edge", prune=True)
                node_list = list(g.nodes.keys())
                mats[model_id] = build_rank_delta_matrix(edges, node_list, granularity="layer")

            # plot_rank_delta_matrices(mats, test_accs, list(range(14)))
            # plot_rank_delta_matrices(mats, test_accs, node_list)
            # W, H_matrices, correlations = analyze_circuits_with_nmf(
            #     mats, test_accs, n_components=5, plot=True
            # )
            # W, H_matrices, correlations = analyze_circuits_with_pls(
            #     mats, test_accs, n_components=5, plot=True
            # )
            W, H_matrices, correlations = analyze_circuits_with_cca(
                mats, test_accs, plot=True, task=task, domain=domain,
            )
            Hs.append(H_matrices)
            all_mats.append(mats)
            all_accs.append(test_accs)

    import pickle
    with open('Hs.p', 'wb') as file:
        pickle.dump((Hs, all_mats, all_accs), file)

def nmf_on_signed_matrices(Hs, n_components=3, plot=True):
    """
    Perform NMF on CCA loadings split into positive and negative parts.

    Parameters
    ----------
    Hs : list of np.ndarray
        List of CCA loading matrices (L x L).
    n_components : int
        Number of motifs to extract.
    plot : bool
        Whether to visualize motifs.

    Returns
    -------
    W : np.ndarray
        Task-specific coefficients (n_tasks x n_components).
    H_motifs : np.ndarray
        Learned motifs reshaped as matrices (n_components x L x L).
    """
    mats_proc = []
    for H in Hs:
        H = np.squeeze(H)
        mats_proc.append(H)

    # Stack (2 * n_tasks, L*L)
    mats_proc = np.stack(mats_proc, axis=0)
    n_cases, L, _ = mats_proc.shape
    X = mats_proc.reshape(n_cases, L * L)

    # Run NMF
    nmf = NMF(n_components=n_components, init="nndsvda", max_iter=500, random_state=42)
    W = nmf.fit_transform(X)
    H_flat = nmf.components_
    H_motifs = H_flat.reshape(n_components, L, L)

    if plot:
        fig, axes = plt.subplots(1, n_components, figsize=(4 * n_components, 4))
        if n_components == 1:
            axes = [axes]
        for i in range(n_components):
            im = axes[i].imshow(H_motifs[i], cmap="viridis")
            axes[i].set_title(f"NMF Motif {i}")
            plt.colorbar(im, ax=axes[i], fraction=0.046, pad=0.04)
        plt.tight_layout()
        plt.savefig("./figures/NMF_signed_motifs.png")

    return W, H_motifs


def multitask_cca_concat(Hs, accs_list, n_components=1, task_names=None, plot=True):
    """
    Concatenated CCA: just pool all tasks into one dataset and run CCA.

    Hs: list[dict], each dict model_id -> (LxL matrix)
    accs_list: list[dict], each dict model_id -> accuracy
    """
    # ---- Flatten all tasks
    X_blocks, y_blocks = [], []
    for H_dict, acc_dict in zip(Hs, accs_list):
        mats = np.stack(list(H_dict.values()), axis=0)
        accs = np.array(list(acc_dict.values()))
        X_blocks.append(mats.reshape(mats.shape[0], -1))
        y_blocks.append(accs.reshape(-1, 1))

    X = np.vstack(X_blocks)
    y = np.vstack(y_blocks)
    X = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-21)  # z-score per-feature
    y = (y - y.mean()) / (y.std() + 1e-21)
    from rcca import CCA
    # ---- Run CCA
    cca = CCA(kernelcca=False, numCC=1, reg=1e-3)
    cca.train([X, y])
    X_c, y_c = cca.comps[0], cca.comps[1]

    # Flip sign if negative correlation globally
    if np.corrcoef(X_c[:, 0], y.ravel())[0, 1] < 0:
        cca.x_weights_[:, 0] *= -1

    # ---- Extract motif
    L = mats.shape[1]
    motif = cca.x_weights_[:, 0].reshape(L, L)

    # ---- Per-task correlations
    corrs = []
    offset = 0
    for t, (Xb, yb) in enumerate(zip(X_blocks, y_blocks)):
        n = Xb.shape[0]
        scores = Xb @ cca.x_weights_[:, 0]
        rho, _ = spearmanr(scores, yb.ravel())
        corrs.append(rho)
        offset += n

    # Plot
    if plot:
        n_tasks = len(Hs)
        ncols = min(3, n_tasks)
        nrows = int(np.ceil(n_tasks / ncols))
        fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows), squeeze=False)
        for t, (Xb, yb) in enumerate(zip(X_blocks, y_blocks)):
            r, c = divmod(t, ncols)
            ax = axes[r, c]
            scores = Xb @ cca.x_weights_[:, 0]
            ax.scatter(scores, yb.ravel(), alpha=0.7)
            title = task_names[t] if task_names else f"Task {t}"
            ax.set_title(f"{title}\nSpearman rho={corrs[t]:.2f}")
        plt.tight_layout()
        plt.savefig('./figures/MultiTask_CCA.png')

    return motif, corrs


def meta_cca(Hs, accs_list, n_iter=1000, lr=1e-2, seed=0, plot=True, task_names=None, outdir=None):
    """
    Meta-CCA: optimize one shared projection w to maximize average Spearman correlation across tasks.

    Parameters
    ----------
    Hs : list[dict]
        Each dict model_id -> adjacency matrix (L×L).
    accs_list : list[dict]
        Each dict model_id -> accuracy (float).
    n_iter : int
        Number of optimization iterations.
    lr : float
        Learning rate for optimizer.
    seed : int
        Random seed for reproducibility.
    plot : bool
        If True, generate motif heatmap and per-task scatter plots.
    task_names : list[str] or None
        Names for each task.
    outdir : str or None
        If provided, save plots to this directory.

    Returns
    -------
    motif : np.ndarray
        Learned universal motif (L×L).
    task_corrs : list[float]
        Spearman correlation per task.
    """

    torch.manual_seed(seed)

    # ---- Preprocess
    X_blocks, y_blocks = [], []
    for H_dict, acc_dict in zip(Hs, accs_list):
        mats = np.stack(list(H_dict.values()), axis=0)
        accs = np.array(list(acc_dict.values()))
        X_blocks.append(torch.tensor(mats.reshape(mats.shape[0], -1), dtype=torch.float32))
        y_blocks.append(torch.tensor(accs, dtype=torch.float32))

    L2 = X_blocks[0].shape[1]
    w = torch.randn(L2, requires_grad=True)

    opt = Adam([w], lr=lr)

    def pearson_corr(x, y):
        x = x - x.mean()
        y = y - y.mean()
        return (x * y).sum() / (torch.norm(x) * torch.norm(y) + 1e-8)

    # ---- Optimize
    for it in range(n_iter):
        opt.zero_grad()
        corrs = []
        for Xb, yb in zip(X_blocks, y_blocks):
            scores = Xb @ w
            rho = pearson_corr(scores, yb)
            corrs.append(rho)
        loss = -torch.stack(corrs).mean()
        loss.backward()
        opt.step()

    # ---- Final projection
    w_final = w.detach().cpu().numpy()
    L = int(np.sqrt(L2))
    motif = w_final.reshape(L, L)

    # ---- Evaluate with true Spearman
    task_corrs = []
    scores_list = []
    for Xb, yb in zip(X_blocks, y_blocks):
        scores = Xb @ torch.tensor(w_final, dtype=torch.float32)
        rho = spearmanr(scores.detach().cpu().numpy(), yb.numpy())[0]
        task_corrs.append(rho)
        scores_list.append((scores.detach().cpu().numpy(), yb.numpy()))

    # ---- 4. Plot
    if plot:
        # Heatmap of motif
        plt.figure(figsize=(5, 4))
        im = plt.imshow(motif, cmap="viridis")
        plt.colorbar(im, fraction=0.046, pad=0.04)
        plt.title("Meta-CCA Universal Motif")
        plt.savefig(f"figures/metaCCA_motif.png", dpi=200)

        # Scatter plots per task
        n_tasks = len(Hs)
        ncols = min(3, n_tasks)
        nrows = int(np.ceil(n_tasks / ncols))
        fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows), squeeze=False)
        for t, (scores, yb) in enumerate(scores_list):
            r, c = divmod(t, ncols)
            ax = axes[r, c]
            ax.scatter(scores, yb, alpha=0.7)
            title = task_names[t] if task_names else f"Task {t}"
            ax.set_title(f"{title}\nSpearman rho={task_corrs[t]:.2f}")
            ax.set_xlabel("Projection score")
            ax.set_ylabel("OOD Accuracy")
            ax.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"figures/metaCCA_scatter.png", dpi=200)

    return motif, task_corrs

meta_groups = {
    'PACS': [],            # all PACS*, PACS-photo, PACS-cartoon, ...
    'terra-incognita': [], # all terra-incognita-*
    'camelyon17': []       # camelyon17 only
}

# Fill groups with (task, domain) pairs
for task, domains in tasks.items():
    if task.startswith("PACS"):
        for d in domains:
            meta_groups["PACS"].append((task, d))
    elif task.startswith("terra-incognita"):
        for d in domains:
            meta_groups["terra-incognita"].append((task, d))
    elif task.startswith("camelyon17"):
        for d in domains:
            meta_groups["camelyon17"].append((task, d))

# Now meta_groups['PACS'] might contain ~12 domains total

Hs_index = 0  # global pointer into Hs[]
taskname2print = {
    'PACS': 'sketch',
    'PACS-photo': 'photo',
    'PACS-art_painting': 'art painting',
    'PACS-cartoon': 'cartoon',
    'terra-incognita-38': 'location 38',
    'terra-incognita-43': 'location 43',
    'terra-incognita-46': 'location 46',
    'terra-incognita-100': 'location 100',
    'camelyon17': 'ID',
}
for meta_task, domain_list in meta_groups.items():

    num_domains = len(domain_list)
    nrows = 2
    ncols = int(np.ceil(num_domains / nrows))

    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(4*ncols, 4*nrows),
        squeeze=False
    )
    axes = axes.flatten()

    for i, (taskname, domain) in enumerate(domain_list):
        ax = axes[i]

        H = Hs[Hs_index][0]
        Hs_index += 1

        im = ax.imshow(H, cmap="viridis")
        ax.set_title(f"{taskname2print[taskname]} → {domain}", fontsize=14)

        # ticks
        num_ticks = H.shape[0]
        ticks = list(range(num_ticks))
        labels = []
        for t in ticks:
            if t == 0:
                labels.append(r"$\mathrm{I}$")
            elif t == num_ticks - 1:
                labels.append(r"$\mathrm{O}$")
            else:
                labels.append(str(t))

        ax.set_xticks(ticks)
        ax.set_xticklabels(labels, fontsize=10, fontfamily='serif')
        ax.set_yticks(ticks)
        ax.set_yticklabels(labels, fontsize=10, fontfamily='serif')
        ax.invert_yaxis()
        ax.set_xlabel("Target Layer", fontsize=13)
        ax.set_ylabel("Source Layer", fontsize=13)

        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # Hide unused subplots
    for k in range(num_domains, nrows*ncols):
        axes[k].axis('off')

    plt.tight_layout()
    plt.savefig(f"./figures/CCAs_pdf_reverse/{meta_task}_all_domains.pdf")
    plt.close(fig)

task_names = []
for task, domains in tasks.items():
    for domain in domains:
        task_names.append(f"{task},{domain}")
# for i, task_name in enumerate(task_names):
#     mats = all_mats[i]
#     analyze_circuits_with_cca_cv(mats, all_accs[i], kcca=True, task=task_name.split(',')[0], domain=task_name.split(',')[1])

for i, task_name in enumerate(task_names):
    fig, ax = plt.subplots(1, 1, figsize=(4, 8))
    im = ax.imshow(Hs[i][0], cmap="viridis")
    ax.set_title(f"CCA Comp {0}")

    num_ticks = Hs[i][0].shape[0]
    ticks = list(range(num_ticks))

    # Custom tick labels with serif font
    labels = []
    for t in ticks:
        if t == 0:
            labels.append(r"$\mathrm{I}$")  # distinguish from 1
        elif t == num_ticks - 1:
            labels.append(r"$\mathrm{O}$")  # distinguish from 0
        else:
            labels.append(str(t))

    ax.set_xticks(ticks)
    ax.set_xticklabels(labels, fontsize=11, fontfamily='serif')
    ax.set_yticks(ticks)
    ax.set_yticklabels(labels, fontsize=11, fontfamily='serif')
    ax.invert_yaxis()

    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.savefig(f'./figures/CCAs_pdf_reverse/{task_name}_viz_node.pdf')
    plt.close(fig)

filtered_mats = []
filtered_accs = []
filtered_names = []

for mat, acc, name in zip(all_mats, all_accs, task_names):
    if "terra" not in name and "camelyon17" not in name:
        filtered_mats.append(mat)
        filtered_accs.append(acc)
        filtered_names.append(name)

# overwrite if you want
# all_mats, all_accs, task_names = filtered_mats, filtered_accs, filtered_names
# comp_matrices, correlations = multitask_cca_concat(
#     all_mats, all_accs, n_components=1, task_names=task_names,
# )
# comp_matrices, correlations = meta_cca(
#     all_mats, all_accs, task_names=task_names,
# )

Hs = np.concatenate(Hs)
Hs_norm = np.zeros_like(Hs)
for i in range(Hs.shape[0]):  # over tasks
    std = Hs[i].std()
    Hs_norm[i] = Hs[i] / (std + 1e-12)
H_mean = Hs_norm.mean(axis=0)
H_var = Hs_norm.var(axis=0)
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 2, figsize=(10, 4))

# Shared tick setup
num_ticks = H_mean.shape[0]
ticks = list(range(num_ticks))
labels = [r"$\mathrm{I}$" if t == 0 else (r"$\mathrm{O}$" if t == num_ticks - 1 else str(t)) for t in ticks]

# === Mean motif heatmap ===
im0 = axes[0].imshow(H_mean, cmap="viridis")
axes[0].set_xticks(ticks)
axes[0].set_xticklabels(labels, fontsize=11, fontfamily='serif')
axes[0].set_yticks(ticks)
axes[0].set_yticklabels(labels, fontsize=11, fontfamily='serif')
axes[0].invert_yaxis()

# === Variance heatmap ===
im1 = axes[1].imshow(H_var, cmap="viridis")
axes[1].set_xticks(ticks)
axes[1].set_xticklabels(labels, fontsize=11, fontfamily='serif')
axes[1].set_yticks(ticks)
axes[1].set_yticklabels(labels, fontsize=11, fontfamily='serif')
axes[1].invert_yaxis()

# Adjust layout before adding colorbars
plt.tight_layout(rect=[0, 0, 1, 0.92])

# Add colorbars ABOVE each subplot
for ax, im in zip(axes, [im0, im1]):
    # Position colorbar manually just above each axes
    bbox = ax.get_position()
    cbar_ax = fig.add_axes([bbox.x0, bbox.y1 + 0.03, bbox.width, 0.03])  # [left, bottom, width, height]
    cbar = fig.colorbar(im, cax=cbar_ax, orientation='horizontal')
    cbar.ax.tick_params(labelsize=9)
    cbar.ax.xaxis.set_ticks_position('top')
    cbar.ax.xaxis.set_label_position('top')

plt.savefig('./figures/all_CCAs_mean_var_matrices_reverse.pdf', bbox_inches='tight')
plt.close(fig)
motif_vec = H_mean.flatten()

for task, domains in tasks.items():
    for domain in domains:
        SWEEP_CSV = f"/home/yxpengcs/PycharmProjects/vit-spurious-robustness/output/{task}_sweep_results_new.csv"
        sweep = pd.read_csv(SWEEP_CSV)

        mats = []
        test_accs = []
        for row_id, row in tqdm(sweep.iterrows()):
            model_id = row.model_id
            if 'terra' in task:
                test_accs.append(row[f'test_f1_{domain}'])
            else:
                test_accs.append(row[f'test_acc_{domain}'])
            if 'location' in domain:
                domain_name = domain.replace('_', '')
            else:
                domain_name = domain

            gpath = (
                f"/home/yxpengcs/PycharmProjects/vMIB-circuit/circuits/EAP-IG-inputs_mean-positional_edge_train_kl_divergence/"
                f"{task.replace('_', '-')}-mean-{domain_name.replace('_', '-')}_sweep_{model_id}/importances.pt"
            )
            per_example_scores_path = gpath.replace("importances.pt", "perexample_importances.p")
            if residual:
                gpath = (
                    f"/home/yxpengcs/PycharmProjects/vMIB-circuit/circuits/EAP-IG-inputs_mean-positional_edge_train_kl_divergence/"
                    f"{task.replace('_', '-')}-mean-{domain_name.replace('_', '-')}_sweep_{model_id}/residual_importances.pt"
                )

            # if not (os.path.exists(gpath) and os.path.exists(per_example_scores_path)):
            #     print(f"[WARN] Missing graph or perexample scores for domain={domain}; skipping.")
            #     continue

            # with open(per_example_scores_path, "rb") as f:
            #     per_example_scores = pickle.load(f)
            edges = {}
            g = Graph.from_pt(gpath)
            for e in g.edges.values():
                edges[f'{e.parent.name}->{e.child.name}<{e.qkv}>'] = e.score.abs()
            # keep same pruning as your previous code
            if residual:
                g.apply_topn(TOPN, True, level="edge", prune=True)  # the negative residual also important
            else:
                g.apply_topn(TOPN, False, level="edge", prune=True)
            node_list = list(g.nodes.keys())
            mats.append(build_rank_delta_matrix(edges, node_list, granularity="layer"))
        scores = []
        for mat in mats:
            mat = mat.flatten()
            score = np.dot(mat, motif_vec)
            scores.append(score)

        scores, accs = np.array(scores), np.array(test_accs)

        # ---- Step 3: correlations ----
        pear, _ = pearsonr(scores, accs)
        spear, _ = spearmanr(scores, accs)

        plt.figure(figsize=(5, 4))
        plt.scatter(scores, accs, alpha=0.7)
        plt.xlabel("Projection on Mean CCA Motif")
        plt.ylabel("OOD Accuracy")
        plt.title(f"r={pear:.2f}, rho={spear:.2f}")
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig("./figures/meanCCA_projection.png")

nmf_on_signed_matrices(Hs)