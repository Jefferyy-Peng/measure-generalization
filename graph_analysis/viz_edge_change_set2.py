import pandas as pd
from matplotlib.colors import TwoSlopeNorm
from scipy.stats import pearsonr, spearmanr
from sklearn.cross_decomposition import CCA
from sklearn.decomposition import NMF
from tqdm import tqdm

from eap.graph import Graph

import numpy as np
import matplotlib.pyplot as plt


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
        plt.savefig('./figures/NMF_viz_set2.png')

    return W, H_matrices, correlations
def analyze_circuits_with_cca(mats, test_accs, n_components=1, plot=True, rank_corr=True, task=''):
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

        # Ensure axes is 2D for consistent indexing
        if n_components == 1:
            axes = axes.reshape(2, 1)

        ticks = list(range(14))
        labels = [r"$\mathrm{I}$" if t == 0 else (r"$\mathrm{O}$" if t == 13 else str(t)) for t in ticks]

        for i in range(n_components):
            ax_h = axes[0, i]
            mat = comp_matrices[i]

            # --- fix colormap centering ---
            vabs = np.abs(mat).max()
            norm = TwoSlopeNorm(vcenter=0, vmin=-vabs, vmax=vabs)
            im = ax_h.imshow(mat, cmap="viridis", norm=norm)

            ax_h.set_xticks(ticks)
            ax_h.set_xticklabels(labels, fontsize=11, fontfamily='serif')
            ax_h.set_yticks(ticks)
            ax_h.set_yticklabels(labels, fontsize=11, fontfamily='serif')
            ax_h.invert_yaxis()
            ax_h.set_title(f"CCA Comp {i}")
            plt.colorbar(im, ax=ax_h, fraction=0.046, pad=0.04)

            ax_s = axes[1, i]
            ax_s.scatter(X_c[:, i], accs, alpha=0.7)
            ax_s.set_xlabel(f"Comp {i} score")
            ax_s.set_ylabel("OOD Acc")

            if rank_corr:
                ax_s.set_title(f"ρ={correlations[i]['spearman_rho']:.2f}")
            else:
                ax_s.set_title(f"r={correlations[i]['pearson_r']:.2f}")

        plt.tight_layout()
        plt.savefig(f'./figures/set2_CCAs_new/{task}_viz_node.pdf')
        plt.close(fig)

    return X_c, comp_matrices, correlations

model_id = 0
residual = False
include_id = True
TOPN = 500
task = 'PACS-set2'
if residual:
    resid_str = "_resid"
else:
    resid_str = ""
SWEEP_CSV = f"/home/yxpengcs/PycharmProjects/vit-spurious-robustness/output/{task}_sweep_results_new.csv"
sweep = pd.read_csv(SWEEP_CSV)

test_cols = sweep.filter(regex=r'^test_f1_').columns.tolist()
domains = [c.replace("test_f1_", "") for c in test_cols]
exclude_domains = ['0', '1', '2', '3', 'photo', 'cartoon', 'art_painting', 'id2', 'time1_region4', 'id2']
for exclude_domain in exclude_domains:
    if exclude_domain in domains:
        domains.remove(exclude_domain)

exclude_strs = ['1','2','3']
new_domains = []
for domain in domains:
    include = True
    for exclude_str in exclude_strs:
        if exclude_str in domain:
            include = False
    if include:
        new_domains.append(domain)
domains = new_domains

circuits = {}
mats = {}
test_accs = {}

for col in sweep.filter(regex=r'^test_f1_').columns:
    domain = col.replace("test_f1_", "")
    if domain in domains:
        test_accs[domain] = sweep[col].iloc[0]  # or .iloc[0] if only one row per domain

ordered_domains = sorted(test_accs.keys(), key=lambda d: test_accs[d], reverse=True)
for domain in tqdm(ordered_domains):
    # ---- Load graph ONCE for this domain ----
    gpath = (
        f"/home/yxpengcs/PycharmProjects/vMIB-circuit/circuits/EAP-IG-inputs_mean-positional_edge_train_kl_divergence/"
        f"{task}-mean-{domain.replace('_', '-')}_sweep_{model_id}/importances.pt"
    )

    per_example_scores_path = gpath.replace("importances.pt", "perexample_importances.p")
    if residual:
        gpath = (
            f"/home/yxpengcs/PycharmProjects/vMIB-circuit/circuits/EAP-IG-inputs_mean-positional_edge_train_kl_divergence/"
            f"{task}-mean-{domain.replace('_', '-')}_sweep_{model_id}/residual_importances.pt"
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
        g.apply_topn(TOPN, True, level="edge", prune=True) # the negative residual also important
    else:
        g.apply_topn(TOPN, False, level="edge", prune=True)
    node_list = list(g.nodes.keys())
    mats[domain] = build_rank_delta_matrix(edges, node_list, granularity="layer")


# plot_rank_delta_matrices(mats, test_accs, list(range(14)))
# plot_rank_delta_matrices(mats, test_accs, node_list)
# W, H_matrices, correlations = analyze_circuits_with_nmf(
#     mats, test_accs, n_components=5, plot=True
# )
W, H_matrices, correlations = analyze_circuits_with_cca(
    mats, test_accs, plot=True, task=task,
)