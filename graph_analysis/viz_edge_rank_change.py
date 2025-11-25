import pandas as pd
from tqdm import tqdm

from eap.graph import Graph

import numpy as np
import matplotlib.pyplot as plt


def build_rank_delta_matrix(edges_id, edges_ood, node_list):
    """
    Build adjacency matrix where entry [i, j] = rank_delta of edge i->j.

    edges_id, edges_ood: dict[(src, dst)] -> importance score
    node_list: ordered list of node names (to fix matrix ordering)
    """
    # Get scores and rank them
    all_edges = list(edges_id.keys() & edges_ood.keys())  # intersection
    id_scores = np.array([edges_id[e].abs().item() for e in all_edges])
    ood_scores = np.array([edges_ood[e].abs().item() for e in all_edges])

    # id_ranks = id_scores.argsort().argsort()  # dense ranks
    # ood_ranks = ood_scores.argsort().argsort()
    # rank_delta = abs(ood_ranks - id_ranks)

    id_ranks = id_scores  # dense ranks
    ood_ranks = ood_scores
    rank_delta = abs(ood_ranks - id_ranks)

    # Map nodes to index
    node_to_idx = {n: i for i, n in enumerate(node_list)}
    n_layers = 14
    mat_layer = np.zeros((n_layers, n_layers))
    # mat = np.zeros((len(node_list), len(node_list)))
    counts = np.zeros((n_layers, n_layers))

    def get_layer(n):
        if n.startswith("a"):
            return int(n.split(".")[0][1:]) + 1
        elif n.startswith("m"):
            return int(n[1:]) + 1
        elif n == "logits":
            return 12 + 1
        else:
            return 0
    for (edge, delta) in zip(all_edges, rank_delta):
        src, dst = edge.split('->')[0], edge.split('->')[1].split('<')[0]
        i, j = get_layer(src), get_layer(dst)
        mat_layer[i, j] += delta
        counts[i, j] += 1

    mat_layer /= counts + 1e-12
    return mat_layer


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
    for idx, domain in enumerate(['hospital0_slide3', 'hospital3_slide34']):
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
    plt.savefig('./figures/rank_change_heatmap_camelyon17-test.png')


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

def rank_change_histograms(circuits, sweep, domains, num_layers=12, interval=1, task="task_name"):
    """
    circuits[domain] = [edges, id_edges]
      where edges: {(src, tgt): score}, id_edges: {(src, tgt): score}
    """
    test_accs = {}
    for col in sweep.filter(regex=r'^test_f1_').columns:
        domain = col.replace("test_f1_", "")
        if domain in domains:
            test_accs[domain] = sweep[col].iloc[0]  # or .iloc[0] if only one row per domain

    # Order domains by accuracy
    ordered_domains = sorted(test_accs.keys(), key=lambda d: test_accs[d], reverse=True)
    domains = list(circuits.keys())
    ncols = len(domains) // interval + 1
    fig, axes = plt.subplots(2, ncols, figsize=(4 * ncols, 6), sharey="row", sharex=True)
    for col, domain in enumerate(ordered_domains[::interval]):
        # --- Compute rank positions for ID and OOD ---
        # sort edges by score (descending rank = important)
        edges, id_edges = circuits[domain]
        id_sorted = sorted(id_edges.items(), key=lambda x: -x[1])
        ood_sorted = sorted(edges.items(), key=lambda x: -x[1])

        id_ranks = {edge: rank for rank, (edge, _) in enumerate(id_sorted)}
        ood_ranks = {edge: rank for rank, (edge, _) in enumerate(ood_sorted)}

        # collect per-layer Δrank
        start_changes = {l: [] for l in range(num_layers+2)}
        end_changes   = {l: [] for l in range(num_layers+2)}

        for edge in id_ranks.keys():
            if edge not in ood_ranks:
                continue
            Δrank = abs(id_ranks[edge] - ood_ranks[edge])

            # --- parse layer index from node name ---
            src = edge.split('->')[0]
            tgt = edge.split('->')[1].split('<')[0]
            # assume names like "a0.h0", "m3", "logits"
            def get_layer(n):
                if n.startswith("a"):
                    return int(n.split(".")[0][1:]) + 1
                elif n.startswith("m"):
                    return int(n[1:]) + 1
                elif n == "logits":
                    return num_layers + 1
                else:
                    return 0

            src_layer = get_layer(src)
            tgt_layer = get_layer(tgt)

            if src_layer is not None and src_layer <= num_layers:
                start_changes[src_layer].append(Δrank)
            if tgt_layer is not None and tgt_layer <= num_layers + 1:
                end_changes[tgt_layer].append(Δrank)

        # --- aggregate to average per bin ---
        avg_start = [np.mean(start_changes[l]) if start_changes[l] else 0
                     for l in range(num_layers+1)]
        avg_end   = [np.mean(end_changes[l]) if end_changes[l] else 0
                     for l in range(1,num_layers+2)]

        # --- plotting ---
        bins = np.arange(num_layers + 1)
        axes[0, col].bar(bins, avg_start, color="steelblue", alpha=0.7)
        axes[1, col].bar(bins+1, avg_end, color="darkorange", alpha=0.7)

        axes[0, col].set_title(f"{domain.replace('val', '1').replace('train', '0').replace('test', '2')}\nAcc={test_accs[domain]:.2f}")
        if col == 0:
            axes[0, col].set_ylabel("Avg ΔRank (by start layer)")
            axes[1, col].set_ylabel("Avg ΔRank (by end layer)")

        axes[1, col].set_xlabel("Layer index")

    plt.tight_layout()
    plt.savefig('./figures/rank_change_hist_camelyon17-test.png')


model_id = 0
residual = False
include_id = True
TOPN = 500
task = 'camelyon17-set2'
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
# exclude_strs = ['1', '2', '3']
# for exclude_str in exclude_strs:
#     domains = [x for x in domains if exclude_str not in x]
# exclude_strs = ['1', '2', '3']
# for exclude_str in exclude_strs:
#     domains = [x for x in domains if exclude_str not in x]
# if include_id:
#     domains += ['id', 'id1', 'id2']

circuits = {}
mats = {}
test_accs = {}
for col in sweep.filter(regex=r'^test_f1_').columns:
    domain = col.replace("test_f1_", "")
    if domain in domains:
        test_accs[domain] = sweep[col].iloc[0]  # or .iloc[0] if only one row per domain

# Order domains by accuracy
ordered_domains = sorted(test_accs.keys(), key=lambda d: test_accs[d], reverse=True)
for domain in tqdm(['hospital3_slide32', 'hospital3_slide30', 'hospital4_slide49']):
    # ---- Load graph ONCE for this domain ----
    gpath = (
        f"/home/yxpengcs/PycharmProjects/vMIB-circuit/circuits/EAP-IG-inputs_mean-positional_edge_train_kl_divergence/"
        f"{task}-mean-{domain.replace('_', '-')}_sweep_{model_id}/importances.pt"
    )
    id_gpath = (
        f"/home/yxpengcs/PycharmProjects/vMIB-circuit/circuits/EAP-IG-inputs_mean-positional_edge_train_kl_divergence/"
        f"{task}-mean-id_sweep_{model_id}/importances.pt"
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
    id_edges = {}
    g = Graph.from_pt(gpath)
    for e in g.edges.values():
        edges[f'{e.parent.name}->{e.child.name}<{e.qkv}>'] = e.score.abs()
    id_g = Graph.from_pt(id_gpath)
    for e in id_g.edges.values():
        id_edges[f'{e.parent.name}->{e.child.name}<{e.qkv}>'] = e.score.abs()
    circuits[domain] = [edges, id_edges]
    # keep same pruning as your previous code
    if residual:
        g.apply_topn(TOPN, True, level="edge", prune=True) # the negative residual also important
    else:
        g.apply_topn(TOPN, False, level="edge", prune=True)
        id_g.apply_topn(TOPN, False, level="edge", prune=True)
    node_list = list(g.nodes.keys())
    mats[domain] = build_rank_delta_matrix(id_edges, edges, node_list)

plot_rank_delta_matrices(mats, test_accs, list(range(14)))
rank_change_histograms(circuits, sweep, domains, task=task, interval=8)