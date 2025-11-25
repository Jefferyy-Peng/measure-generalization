import argparse
from copy import copy, deepcopy
import pickle
import re
from collections import defaultdict
from typing import Dict, Literal, Tuple
from torch.nn import functional as F

from wilds.common.metrics.all_metrics import Accuracy, Recall, F1
import pandas as pd
from scipy.stats import spearmanr, pearsonr
import timm
from scipy.stats import norm, kendalltau, spearmanr
from tqdm import tqdm

import numpy as np
from sklearn.linear_model import LinearRegression
import torch
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import Ridge, Lasso
import networkx as nx

from dataset_utils import get_metashift_dataset, get_loader
from eap.graph import Graph, AttentionNode, MLPNode
import os
import math
from eap.visualization import generate_random_color, get_color

import networkx as nx
from networkx.algorithms.isomorphism import DiGraphMatcher
import pygraphviz as pgv

# ---------------------------------------
# Graph loading with memo-cache
# ---------------------------------------
GRAPH_DIR = "circuits/EAP-IG-inputs_mean-positional_edge_train_kl_divergence"
_GRAPH_CACHE: Dict[int, Graph] = {}
task = 'PACS-photo'
split = 'sketch'
circuit_type = '_kl'

def load_graph(model_id: int):
    if model_id in _GRAPH_CACHE:
        return _GRAPH_CACHE[model_id]
    gpath = f"{GRAPH_DIR}/{task.replace('_', '-')}-mean-{split.replace('_','-')}_sweep_{model_id}/importances.pt"
    per_example_scores_path = gpath.replace("importances.pt", "perexample_importances.p")
    try:
        with open(per_example_scores_path, 'rb') as file:
            per_example_scores = pickle.load(file)
    except:
        per_example_scores = None
    try:
        ob_auc_path = gpath.replace("importances.pt", "ob_auc_100.pt")
        with open(ob_auc_path, 'rb') as file:
            ob_auc = pickle.load(file)
        bg_auc_path = gpath.replace("importances.pt", "bg_auc_100.pt")
        with open(bg_auc_path, 'rb') as file:
            bg_auc = pickle.load(file)
    except:
        ob_auc = None
        bg_auc = None
    g = Graph.from_pt(gpath)
    g.apply_topn(200, False, level="edge", prune=True)
    _GRAPH_CACHE[model_id] = (g, per_example_scores, ob_auc, bg_auc)
    return g, per_example_scores, ob_auc, bg_auc

def get_layer_index(name: str) -> int:
    """
    Extracts the layer index from a node name.
    Examples:
        'a6.h4' -> 6
        'm2' -> 2
        'input' or 'logits' -> -1
    """
    if name in ['input', 'logits']:
        return -1
    match = re.search(r'[am](\d+)', name)
    if match:
        return int(match.group(1))
    return -1

def flatten_metric_output(metric_name, output):
    if isinstance(output, dict):
        return output
    else:
        return {metric_name: output}

def to_nx_digraph(custom_graph) -> nx.DiGraph:
    """
    Convert a custom transformer-style Graph object to a networkx.DiGraph.
    """
    G = nx.DiGraph()
    for node in custom_graph.nodes.values():
        if node.in_graph:
            G.add_node(node.name)  # You can also add attributes like type=node.type if needed
    for edge in custom_graph.edges.values():
        if edge.in_graph:
            G.add_edge(edge.parent.name, edge.child.name)
    return G

def build_layer_map(nodes):
    layer_map = {}
    for name in nodes:
        if name == "input":
            layer = 0
        elif name.startswith("a"):
            # Match attention node like "a0.h0"
            match = re.match(r"a(\d+)\.h\d+", name)
            if match:
                block_idx = int(match.group(1))
                layer = 1 + block_idx * 2  # attention layer for block i is 1 + 2i
            else:
                raise ValueError(f"Invalid attention node name: {name}")
        elif name.startswith("m"):
            # Match MLP node like "m0"
            match = re.match(r"m(\d+)", name)
            if match:
                block_idx = int(match.group(1))
                layer = 2 + block_idx * 2  # MLP layer for block i is 2 + 2i
            else:
                raise ValueError(f"Invalid MLP node name: {name}")
        elif name == "logits":
            layer = 25
        else:
            raise ValueError(f"Unknown node type: {name}")

        layer_map[name] = layer

    return layer_map


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

def get_layer_from_node_name(name):
    if name == 'input':
        return 0
    elif name.startswith('a'):
        layer_idx = int(name.split('.')[0][1:])
        return 2 * layer_idx + 1
    elif name.startswith('m'):
        layer_idx = int(name[1:])
        return 2 * layer_idx + 2
    elif name.startswith('logits'):
        return 25
    else:
        raise ValueError(f"Unknown node name: {name}")

# ---------------------------------------
# Metric functions
# ---------------------------------------

def _layer_id(name):
    # map "m3" -> 3; "a0.h7" -> 0 (use parent block); "logits" -> big number
    if name == "logits": return 10_000
    m = re.search(r"[ma](\d+)", name)
    return int(m.group(1)) if m else -1

def _jump(e):
    return max(_layer_id(e.child.name) - _layer_id(e.parent.name), 0)

def compute_normed_circuit_instability(graph, per_example_scores):
    edge_indices = torch.tensor(
        [edge.matrix_index for edge in graph.edges.values()]
    ).T
    # Stack all circuits into [N, num_edges] matrix
    all_scores = torch.stack([
        score_tensor[edge_indices[0], edge_indices[1]]
        for score_tensor in per_example_scores['scores']
    ])  # shape: [num_examples, num_edges]

    norm = all_scores.sum(dim=1, keepdim=True).clamp(min=1e-8)
    all_scores = all_scores / norm

    return all_scores.var(dim=0, unbiased=False).mean().item()

def get_num_edges(graph):
    num_edge = 0
    for edge in graph.edges.values():
        if edge.in_graph:
            num_edge += 1

    return num_edge

def get_shortcut_vs_deep_ratio(graph, shallow_thresh=3, deep_thresh=9):
    """
    Compute ratio of shortcut (shallow→deep) edge importance
    vs. local/no-skip edges (within shallow or within deep, + late→late edges).

    Args:
        graph: Graph object with edges having attributes:
               - parent.name (e.g., "a11.h5", "m10")
               - child.name (destination node)
               - score (edge importance value)
        shallow_thresh (int): maximum layer index considered shallow
        deep_thresh (int): minimum layer index considered deep

    Returns:
        float: ratio = shortcut_score / (local_score + 1e-12)
    """

    shortcut_score, deep_score = 0.0, 0.0

    def get_layer(n):
        if n.startswith('a'):
            return int(n.split('.')[0][1:])
        elif n.startswith('m'):
            return int(n[1:])
        elif n == 'input':
            return None
        elif n.startswith('logits'):
            return 14
        else:
            return None

    for edge in graph.edges.values():
        src, dst = edge.parent.name, edge.child.name
        src_layer, dst_layer = get_layer(src), get_layer(dst)
        if src_layer is None or dst_layer is None:
            continue

        score_val = abs(edge.score).item() if hasattr(edge.score, "item") else abs(edge.score)

        # Case 1: Shortcut (shallow -> deep)
        if src_layer <= shallow_thresh and dst_layer >= deep_thresh:
            shortcut_score += score_val

        # Case 2: deep connection (within shallow, within deep, or deep→logits)
        elif src_layer >= deep_thresh and dst_layer >= deep_thresh:
            deep_score += score_val

    return deep_score / (shortcut_score + 1e-12)

def get_edge_start_ratio_deep_vs_shallow(graph, shallow_thresh=0, deep_thresh=11):
    """
    Compute ratio of total edge importance from deep vs. shallow layers.

    Args:
        graph: Graph object with edges that have attributes:
               - parent.name (e.g., "a11.h5", "m10")
               - score (edge importance value)
        shallow_thresh (int): maximum layer index to be considered shallow
        deep_thresh (int): minimum layer index to be considered deep

    Returns:
        float: ratio = deep_score / (shallow_score + 1e-12)
    """
    shallow_score, deep_score = 0.0, 0.0

    for edge in graph.edges.values():
        src = edge.parent.name
        # Handle attention nodes "a{layer}.{head}" or MLP nodes "m{layer}"
        if src.startswith('a'):
            layer = int(src.split('.')[0][1:])
        elif src.startswith('m'):
            layer = int(src[1:])
        elif src == 'input':
            continue
            # layer = 0
        else:
            continue

        score_val = abs(edge.score).item() if hasattr(edge.score, "item") else abs(edge.score)

        if layer <= shallow_thresh:
            shallow_score += score_val
        elif layer >= deep_thresh:
            deep_score += score_val

    return deep_score / (shallow_score + 1e-12)


def get_logit_contribution_ratio_deep_vs_shallow(graph, shallow_thresh=3, deep_thresh=9):
    shallow_score, deep_score = 0.0, 0.0

    for edge in graph.edges.values():
        if edge.child.name == 'logits':
            src = edge.parent.name
            if src.startswith('a'):
                layer = int(src.split('.')[0][1:])
            elif src.startswith('m'):
                layer = int(src[1:])
            else:
                continue
            if layer <= shallow_thresh:
                shallow_score += abs(edge.score).item()
            elif layer >= deep_thresh:
                deep_score += abs(edge.score).item()

    return deep_score / (shallow_score + 1e-12)

def get_generalization_graph_metrics(graph):
    import networkx as nx
    import numpy as np
    from collections import defaultdict
    from networkx.algorithms.community import greedy_modularity_communities
    from scipy.stats import entropy

    # Convert EAP graph to networkx DiGraph
    G = nx.DiGraph()
    edge_weights = {}
    for edge in graph.edges.values():
        if edge.in_graph:
            u = edge.parent.name
            v = edge.child.name
            G.add_edge(u, v)
            edge_weights[(u, v)] = abs(edge.score.item() if isinstance(edge.score, torch.Tensor) else edge.score)

    UG = G.to_undirected()
    nx.set_edge_attributes(G, edge_weights, 'weight')
    nx.set_edge_attributes(UG, edge_weights, 'weight')

    metrics = {}

    # Laplacian spectrum (undirected)
    L = nx.laplacian_matrix(UG, weight='weight').toarray()
    import numpy as np
    import scipy as sp
    sp.errstate = np.errstate
    Lnorm = nx.normalized_laplacian_matrix(UG, weight='weight').toarray()
    eigvals = np.sort(np.real(np.linalg.eigvals(L)))
    normeigvals = np.linalg.eigvalsh(Lnorm)
    d_avg = 2 * UG.number_of_edges() / UG.number_of_nodes()
    metrics['algebraic_connectivity'] = eigvals[1].item() if len(eigvals) > 1 else 0.0
    metrics['spectral_gap'] = (eigvals[1] - eigvals[2]).item() if len(eigvals) > 2 else 0.0
    metrics['graph_energy'] = np.sum(eigvals ** 2).item()
    metrics['laplacian_energy'] = np.sum((eigvals - d_avg)**2).item()
    metrics['normed_laplacian_energy'] = np.sum((normeigvals - 1)**2).item()
    metrics['laplacian_spectral_entropy'] = entropy(np.abs(eigvals / np.sum(np.abs(eigvals)) + 1e-8)).item()

    # Modularity
    try:
        communities = list(greedy_modularity_communities(UG, weight='weight'))
        metrics['modularity'] = nx.algorithms.community.quality.modularity(UG, communities, weight='weight')
        sizes = [len(c) for c in communities]
        metrics['community_size_variance'] = np.var(sizes).item()
    except Exception:
        metrics['modularity'] = np.nan
        metrics['community_size_variance'] = np.nan

    # Clustering
    try:
        metrics['avg_clustering'] = nx.average_clustering(UG, weight='weight')
    except Exception:
        metrics['avg_clustering'] = np.nan

    # Kirchhoff index (effective resistance)
    try:
        from networkx.linalg.laplacianmatrix import laplacian_matrix
        L = laplacian_matrix(UG, weight='weight').toarray()
        L_pinv = np.linalg.pinv(L)
        n = L.shape[0]
        kirchhoff = n * np.trace(L_pinv).item()
        metrics['kirchhoff_index'] = kirchhoff
    except Exception:
        metrics['kirchhoff_index'] = np.nan

    # Average path length
    if nx.is_connected(UG):
        try:
            metrics['avg_path_length'] = nx.average_shortest_path_length(UG, weight='weight')
        except Exception:
            metrics['avg_path_length'] = np.nan
    else:
        metrics['avg_path_length'] = np.nan

    # Path redundancy (simple paths input → logits)
    if 'input' in G and 'logits' in G:
        try:
            metrics['path_redundancy'] = sum(1 for _ in nx.all_simple_paths(G, source='input', target='logits'))
        except Exception:
            metrics['path_redundancy'] = 0
    else:
        metrics['path_redundancy'] = 0

    # Effective path depth
    try:
        depths = nx.single_source_shortest_path_length(G, 'input')
        total_weight = 0.0
        weighted_sum = 0.0
        for u, v in G.edges():
            d = depths.get(u, None)
            if d is not None:
                w = edge_weights.get((u, v), 0.0)
                weighted_sum += d * w
                total_weight += w
        metrics['effective_path_depth'] = weighted_sum / total_weight if total_weight > 0 else 0.0
    except Exception:
        metrics['effective_path_depth'] = 0.0

    # Node importance entropy
    node_scores = defaultdict(float)
    for (u, v), w in edge_weights.items():
        node_scores[u] += w
    values = np.array(list(node_scores.values()))
    probs = values / values.sum() if values.sum() > 0 else np.ones_like(values) / len(values)
    metrics['node_importance_entropy'] = entropy(probs).item()

    # Betweenness centrality (avg)
    try:
        bc = nx.betweenness_centrality(G, weight='weight')
        metrics['avg_betweenness'] = np.mean(list(bc.values())).item()
    except Exception:
        metrics['avg_betweenness'] = np.nan

    # spectral radius
    if not nx.is_connected(UG):
        UG = max((UG.subgraph(c) for c in nx.connected_components(UG)), key=len)
    L = nx.laplacian_matrix(UG, weight='weight').toarray()
    eigenvalues = np.linalg.eigvals(L)
    metrics['spectral_radius'] = np.max(np.abs(eigenvalues)).item()

    # Flow centrality (edge-based)
    try:
        flow_centrality = nx.edge_betweenness_centrality(G, weight='weight')
        metrics['avg_edge_flow_centrality'] = np.mean(list(flow_centrality.values())).item()
    except Exception:
        metrics['avg_edge_flow_centrality'] = np.nan

    # Edge score Gini
    try:
        scores = np.array(list(edge_weights.values()))
        n = len(scores)
        diffsum = np.sum(np.abs(scores[:, None] - scores[None, :]))
        gini = diffsum / (2 * n * scores.sum()).item() if scores.sum() > 0 else np.nan
        metrics['edge_score_gini'] = gini
    except Exception:
        metrics['edge_score_gini'] = np.nan

    return metrics


def _nx_undirected_from_signed_edges(edges_signed):
    """edges_signed: dict[(u,v)] -> w (can be +/-) ; undirected sum."""
    UG = nx.Graph()
    for (u, v), w in edges_signed.items():
        if u == v:
            continue
        if UG.has_edge(u, v):
            UG[u][v]['weight'] += float(w)
        else:
            UG.add_edge(u, v, weight=float(w))
    return UG

def _signed_laplacian(UG_signed):
    """Signed Laplacian Ls = D - A_sigma, with D built from |w|, undirected."""
    nodes = list(UG_signed.nodes())
    idx = {n: i for i, n in enumerate(nodes)}
    n = len(nodes)
    A = np.zeros((n, n), dtype=float)
    for u, v, data in UG_signed.edges(data=True):
        i, j = idx[u], idx[v]
        w = float(data['weight'])  # signed
        A[i, j] = A[j, i] = w
    D = np.diag(np.sum(np.abs(A), axis=1))
    Ls = D - A                       # PSD for undirected signed graphs
    return Ls, nodes

def _avg_shortest_path_len_abs(UG_abs):
    """Shortest paths on inverse absolute weights (stronger edge = shorter)."""
    if not nx.is_connected(UG_abs):
        # take largest component to avoid inf
        UG_abs = max((UG_abs.subgraph(c) for c in nx.connected_components(UG_abs)), key=len)
    invw = {(u, v): 1.0 / max(data['weight'], 1e-12) for u, v, data in UG_abs.edges(data=True)}
    nx.set_edge_attributes(UG_abs, invw, 'invw')
    return nx.average_shortest_path_length(UG_abs, weight='invw')
from scipy.stats import entropy


def _to_float(x):
    return float(x.item()) if hasattr(x, "item") else float(x)

def _edge_key(e, include_qkv=True):
    qkv = getattr(e, "qkv", None) if include_qkv else None
    return (e.parent.name, e.child.name, qkv)

def _edge_score_dict(graph, *, in_graph_only=True, include_qkv=True, use_abs=False):
    d = {}
    for e in graph.edges.values():
        if in_graph_only and not getattr(e, "in_graph", True):
            continue
        try:
            s = _to_float(e.score)
        except Exception:
            continue
        if use_abs:
            s = abs(s)
        d[_edge_key(e, include_qkv=include_qkv)] = s
    return d

def _vector_metrics(x, y, *, p=2):
    """Return SRCC, Pearson, cosine, and residual norms for two numpy arrays."""

    # Spearman
    srcc = float(spearmanr(x, y).correlation)
    # Pearson
    pear = float(pearsonr(x, y).statistic) if hasattr(pearsonr(x, y), "statistic") else float(pearsonr(x, y)[0])
    # Cosine
    denom = (np.linalg.norm(x) * np.linalg.norm(y) + 1e-12)
    cosine = float(np.dot(x, y) / denom)
    # Residual norms
    resid = x - y
    res_p   = float(np.linalg.norm(resid, ord=p))
    res_inf = float(np.linalg.norm(resid, ord=np.inf))

    # Kendall's tau (tau-a) and tau-b (tie-adjusted)
    tau_c = float(kendalltau(x, y, nan_policy='omit', variant='c').correlation)
    tau_b = float(kendalltau(x, y, nan_policy='omit', variant='b').correlation)
    return {
        "srcc": srcc,
        "pearson": pear,
        "cosine": cosine,
        "residual_norm_L{}".format(p): res_p,
        "residual_norm_Linf": res_inf,
        "kendall_tau_c": tau_c,  # tau-a (no tie correction)
        "kendall_tau_b": tau_b,  # tau-b (tie-adjusted)
    }

def _weighted_jaccard(d1, d2):
    keys = set(d1) | set(d2)
    if not keys: return 1.0
    num = sum(min(abs(d1.get(k,0.0)), abs(d2.get(k,0.0))) for k in keys)
    den = sum(max(abs(d1.get(k,0.0)), abs(d2.get(k,0.0))) for k in keys) + 1e-12
    return float(num / den)

def _binary_jaccard(d1, d2, thr=0.0):
    s1 = {k for k, v in d1.items() if abs(v) > thr}
    s2 = {k for k, v in d2.items() if abs(v) > thr}
    if not (s1 or s2): return 1.0
    return float(len(s1 & s2) / (len(s1 | s2)))

def _UG_from_scores(d, collapse_qkv=True, use_abs=True):
    """
    Build an undirected weighted graph from an edge-score dict.
    Keys are (u,v,qkv). If collapse_qkv=True we sum across q/k/v.
    """
    UG = nx.Graph()
    for (u, v, qkv), w in d.items():
        if use_abs: w = abs(w)
        if collapse_qkv:
            key = (u, v)
        else:
            key = (u, f"{v}|{qkv}")
        a, b = key
        if a == b:
            continue
        if UG.has_edge(a, b):
            UG[a][b]["weight"] += w
        else:
            UG.add_edge(a, b, weight=w)
    return UG

def _lap_spec(UG):
    """Normalized Laplacian eigenvalues sorted ascending."""
    if UG.number_of_nodes() < 2 or UG.number_of_edges() == 0:
        return np.array([0.0, 2.0])
    L = nx.normalized_laplacian_matrix(UG, weight="weight").toarray()
    vals = np.linalg.eigvalsh(L)
    vals.sort()
    return vals

def _aspl_abs(UG):
    """Average shortest path using 1/weight as length."""
    if UG.number_of_nodes() < 2 or UG.number_of_edges() == 0:
        return np.nan
    if not nx.is_connected(UG):
        UG = max((UG.subgraph(c) for c in nx.connected_components(UG)), key=len)
    invw = {(u, v): 1.0 / max(d["weight"], 1e-12) for u, v, d in UG.edges(data=True)}
    nx.set_edge_attributes(UG, invw, "invw")
    return float(nx.average_shortest_path_length(UG, weight="invw"))

def _algebraic_connectivity(UG):
    vals = _lap_spec(UG)
    return float(vals[1]) if len(vals) > 1 else 0.0

def _spectral_radius_laplacian(UG):
    if UG.number_of_nodes() < 2 or UG.number_of_edges() == 0:
        return 0.0
    L = nx.laplacian_matrix(UG, weight="weight").toarray()
    return float(np.max(np.abs(np.linalg.eigvals(L))))

def mcs_approx_distance(G1: nx.Graph, G2: nx.Graph, timeout: float = 5.0) -> float:
    """
    Approximate MCS-based distance in [0,1], smaller = more similar.
    We estimate MCS size from an approximate Graph Edit Distance (unit costs).
    GED ~= |V1|+|V2|-2|V_mcs| + |E1|+|E2|-2|E_mcs|
    We recover |E_mcs| component and normalize.
    """
    try:
        ged = nx.graph_edit_distance(G1, G2, timeout=timeout)
        if ged is None:
            return float("nan")
        n1, n2 = G1.number_of_nodes(), G2.number_of_nodes()
        e1, e2 = G1.number_of_edges(), G2.number_of_edges()
        # heuristic: recover a single "effective" MCS size from edges (more informative)
        e_mcs = max(0.0, (e1 + e2 - float(ged)) / 2.0)
        # normalize distance by the larger edge count; if both 0, fall back to nodes
        if (e1 + e2) > 0:
            d = 1.0 - (2.0 * e_mcs) / (e1 + e2)
        else:
            v_mcs = max(0.0, (n1 + n2 - float(ged)) / 2.0)
            d = 1.0 - (2.0 * v_mcs) / (n1 + n2) if (n1 + n2) > 0 else 0.0
        return float(max(0.0, min(1.0, d)))
    except Exception:
        return float("nan")


# -------------------------------
# 2) Laplacian spectral distance
# -------------------------------
def laplacian_spectral_distance(G1: nx.Graph, G2: nx.Graph, k: int = 50, normalized: bool = True) -> float:
    """
    L2 distance between the smallest k Laplacian eigenvalues (padded with zeros).
    """
    def eigs(G):
        if normalized:
            L = nx.normalized_laplacian_matrix(G).astype(float).toarray()
        else:
            L = nx.laplacian_matrix(G).astype(float).toarray()
        # eigh for symmetric matrices
        w = np.linalg.eigvalsh(L)
        w = np.sort(np.real(w))
        return w

    w1, w2 = eigs(G1), eigs(G2)
    # take first k (smallest), pad to length k
    s1 = np.pad(w1[:k], (0, max(0, k - len(w1))), constant_values=0.0)
    s2 = np.pad(w2[:k], (0, max(0, k - len(w2))), constant_values=0.0)
    return float(np.linalg.norm(s1 - s2))


# -------------------------------
# 3) NetLSD (heat trace) distance
# -------------------------------
def netlsd_distance(G1: nx.Graph, G2: nx.Graph, num_scales: int = 250, t_min: float = 1e-2, t_max: float = 1e2) -> float:
    """
    Compute NetLSD signatures via heat trace: h_G(t) = sum_i exp(-t * lambda_i),
    where lambda_i are normalized Laplacian eigenvalues. L2 distance between signatures.
    """
    from netlsd import heat

    s1 = heat(G1)
    s2 = heat(G2)
    # same grid, direct L2
    return float(np.linalg.norm(s1 - s2))


# -------------------------------
# 5) Motif counts distance (basic triads/4-cycles)
# -------------------------------
def motif_counts_distance(G1: nx.Graph, G2: nx.Graph) -> float:
    """
    Compare small motif counts (triangles, wedges, squares as 4-cycles).
    Returns L2 distance between the motif-count vectors (log1p-scaled).
    """
    def counts(G):
        # wedges (open triads): sum over nodes C(deg, 2) - 3*triangles_at_node
        deg = dict(G.degree())
        wedges = sum(d * (d - 1) // 2 for d in deg.values()) - sum(nx.triangles(G).values())
        # 4-cycles (squares): approximate via cycle_basis (on simple cycles) – not exact for all 4-cycles,
        # but acceptable as a lightweight proxy
        try:
            # Convert to undirected simple cycles
            cb = nx.cycle_basis(G)
            squares = sum(1 for cyc in cb if len(cyc) == 4)
        except Exception:
            squares = 0
        return np.array([wedges, squares], dtype=float)

    v1, v2 = counts(G1), counts(G2)
    return float(np.linalg.norm(np.log1p(v1) - np.log1p(v2)))

def _degree_array(
    G: nx.Graph,
    directed_mode: Literal["undirected", "in", "out", "total"] = "undirected"
) -> np.ndarray:
    """Return degrees as a 1D numpy array according to directed_mode."""
    if G.number_of_nodes() == 0:
        return np.array([], dtype=float)

    if G.is_directed():
        if directed_mode == "in":
            degs = [d for _, d in G.in_degree()]
        elif directed_mode == "out":
            degs = [d for _, d in G.out_degree()]
        elif directed_mode == "total":
            degs = [G.in_degree(n) + G.out_degree(n) for n in G.nodes()]
        else:  # treat as undirected by symmetrizing degrees
            H = nx.Graph(G)  # ignore direction for degrees
            degs = [d for _, d in H.degree()]
    else:
        degs = [d for _, d in G.degree()]
    return np.asarray(degs, dtype=float)


def _hist_from_degrees(
    d1: np.ndarray, d2: np.ndarray, max_bin: int | None = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Build normalized histograms on a shared support [0..K].
    Returns: bins, p1, p2 where sum(p1)=sum(p2)=1 (unless empty).
    """
    if d1.size == 0 and d2.size == 0:
        return np.arange(1), np.array([0.0]), np.array([0.0])

    K = int(max(np.max(d1) if d1.size else 0, np.max(d2) if d2.size else 0))
    if max_bin is not None:
        K = min(K, int(max_bin))
    bins = np.arange(K + 2) - 0.5  # bin for each integer degree

    h1, _ = np.histogram(d1, bins=bins)
    h2, _ = np.histogram(d2, bins=bins)

    p1 = h1.astype(float); p2 = h2.astype(float)
    s1 = p1.sum(); s2 = p2.sum()
    if s1 > 0: p1 /= s1
    if s2 > 0: p2 /= s2

    centers = np.arange(K + 1)  # 0..K
    return centers, p1, p2


def degree_distribution_distance(
    G1: nx.Graph,
    G2: nx.Graph,
    method: Literal["l1", "l2", "chi2", "jsd", "ks"] = "l2",
    directed_mode: Literal["undirected", "in", "out", "total"] = "undirected",
    max_bin: int | None = None,
    eps: float = 1e-12,
) -> float:
    """
    Distance between degree distributions of two graphs.

    Args:
        method:
            - "l1": L1 distance between normalized histograms
            - "l2": L2 distance between normalized histograms
            - "chi2": Pearson chi-square (symmetrized)
            - "jsd": Jensen–Shannon distance (sqrt of JS divergence, base-2)
            - "ks": Kolmogorov–Smirnov distance between ECDFs
        directed_mode: for directed graphs: "in", "out", "total", or treat as "undirected"
        max_bin: cap maximum degree bin to limit histogram length (optional)
    """
    d1 = _degree_array(G1, directed_mode)
    d2 = _degree_array(G2, directed_mode)

    # Handle empty cases
    if d1.size == 0 and d2.size == 0:
        return 0.0
    if method == "ks":
        # KS on empirical CDFs over shared support
        if d1.size == 0 or d2.size == 0:
            return 1.0
        xs = np.unique(np.concatenate([d1, d2]))
        xs.sort()
        F1 = np.searchsorted(np.sort(d1), xs, side="right") / max(1, len(d1))
        F2 = np.searchsorted(np.sort(d2), xs, side="right") / max(1, len(d2))
        return float(np.max(np.abs(F1 - F2)))

    # Histogram-based methods
    _, p1, p2 = _hist_from_degrees(d1, d2, max_bin=max_bin)

    if method == "l1":
        return float(np.sum(np.abs(p1 - p2)))
    elif method == "l2":
        return float(np.linalg.norm(p1 - p2))
    elif method == "chi2":
        # symmetrized chi-square
        denom = p1 + p2 + eps
        return float(0.5 * np.sum((p1 - p2) ** 2 / denom))
    elif method == "jsd":
        # Jensen–Shannon distance (sqrt(JS divergence))
        m = 0.5 * (p1 + p2) + eps
        p1e = p1 + eps; p2e = p2 + eps
        kl1 = np.sum(p1e * (np.log2(p1e) - np.log2(m)))
        kl2 = np.sum(p2e * (np.log2(p2e) - np.log2(m)))
        jsd = 0.5 * (kl1 + kl2)
        return float(np.sqrt(max(0.0, jsd)))
    else:
        raise ValueError(f"Unknown method: {method}")

# ---- main API ----------------------------------------------------------------
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

        mat /= counts + 1e-12
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
def _filter_constant_entries(v1, v2, tol=1e-22):
    """
    Remove indices where both vectors are always ~0
    to avoid distorting Pearson/Spearman or vector norms.
    """
    mask = ~((np.abs(v1) < tol) & (np.abs(v2) < tol))
    return v1[mask], v2[mask]
def get_layer_distance_from_id(
    g1, g2,
    granularity="layer",
    p_residual=2
):
    """
    Compute distance between two graphs at the layer-wise (or node-wise) level,
    using build_rank_delta_matrix to collapse edge scores.

    Args:
        g1, g2: Graph objects
        node_list: list of node names (for node granularity)
        granularity: "layer" or "node"
        p_residual: p-norm to use for residual vector distance

    Returns:
        dict of distance metrics
    """

    # 1. Build adjacency matrices at chosen granularity
    edges1 = {f"{e.parent.name}->{e.child.name}<{e.qkv}>": e.score for e in g1.edges.values()}
    edges2 = {f"{e.parent.name}->{e.child.name}<{e.qkv}>": e.score for e in g2.edges.values()}

    node_list = list(g1.nodes.keys())

    M1 = build_rank_delta_matrix(edges1, node_list, granularity=granularity)
    M2 = build_rank_delta_matrix(edges2, node_list, granularity=granularity)

    # 2. Flatten to vectors
    v1 = M1.flatten()
    v2 = M2.flatten()
    v1, v2 = _filter_constant_entries(v1, v2)
    # 3. Vector distances
    diff = v1 - v2
    vec_dists = {
        "layer_vec_L1": float(np.sum(np.abs(diff))),
        "layer_vec_L2": float(np.linalg.norm(diff, ord=2)),
        "layer_vec_Linf": float(np.linalg.norm(diff, ord=np.inf)),
        "layer_vec_Lp": float(np.linalg.norm(diff, ord=p_residual)),
    }

    # 4. Correlations
    pear, _ = pearsonr(v1, v2)
    spear, _ = spearmanr(v1, v2)
    corr_dists = {
        "layer_pearson": pear,
        "layer_spearman": spear,
    }

    # 5. Structural / spectral at layer-level
    # (we can treat M1, M2 as adjacency and build undirected graphs)
    UG1 = nx.from_numpy_array(np.abs(M1))
    UG2 = nx.from_numpy_array(np.abs(M2))

    s1 = _lap_spec(UG1); s2 = _lap_spec(UG2)
    k = min(len(s1), len(s2), 10)
    spec_L2 = float(np.linalg.norm(s1[:k] - s2[:k], ord=2))
    spec_Linf = float(np.linalg.norm(s1[:k] - s2[:k], ord=np.inf))

    out = {
        **vec_dists,
        **corr_dists,
        "layer_spec_L2": spec_L2,
        "layer_spec_Linf": spec_Linf,
    }

    return out

def get_distance_from_id(
    g1, g2,
    *,
    in_graph_only=True,
    include_qkv=True,
    collapse_qkv_for_structure=False,   # sum q/k/v when building UG
    use_abs_for_structure=True,        # structure built from |weights|
    k_eigs=10,                         # compare top-k eigenvalues of N-Laplacian
    jaccard_threshold=0.0,
    p_residual=2,
    approximate_ged=True,
    ged_threshold=0.0,
    ged_timeout=2.0
):
    """
    Compute vector, structural, and spectral distances between two EAP IG graphs.
    Returns a flat dict of metrics.
    """

    # --- Edge score vectors (signed) ---
    d1 = _edge_score_dict(g1, in_graph_only=in_graph_only, include_qkv=include_qkv, use_abs=False)
    d2 = _edge_score_dict(g2, in_graph_only=in_graph_only, include_qkv=include_qkv, use_abs=False)

    id_edges = np.array([e.score for e in g1.edges.values()], dtype=float)
    edges = np.array([e.score for e in g2.edges.values()], dtype=float)
    vec = _vector_metrics(id_edges, edges, p=p_residual)

    # also on absolute scores (optional but often useful)
    id_edges_abs = np.array([abs(e.score) for e in g1.edges.values()], dtype=float)
    edges_abs = np.array([abs(e.score) for e in g2.edges.values()], dtype=float)
    vec_abs = _vector_metrics(id_edges_abs, edges_abs, p=p_residual)
    vec_abs = {f"abs_{k}": v for k, v in vec_abs.items()}

    # Jaccard similarities
    bj = _binary_jaccard(d1, d2, thr=jaccard_threshold)

    # --- Structural / spectral comparisons on undirected |w| graphs ---
    UG1 = _UG_from_scores(d1, collapse_qkv=collapse_qkv_for_structure, use_abs=use_abs_for_structure)
    UG2 = _UG_from_scores(d2, collapse_qkv=collapse_qkv_for_structure, use_abs=use_abs_for_structure)

    # Spectra (normalized Laplacian)
    s1 = _lap_spec(UG1); s2 = _lap_spec(UG2)
    k = int(min(k_eigs, len(s1), len(s2)))
    spec_L2 = float(np.linalg.norm(s1[:k] - s2[:k], ord=2))
    spec_Linf = float(np.linalg.norm(s1[:k] - s2[:k], ord=np.inf))

    # Degree distributions (normalized by total weight)
    def norm_deg_vec(UG):
        if UG.number_of_nodes() == 0: return np.zeros(1)
        degs = np.array([d for _, d in UG.degree(weight="weight")], dtype=float)
        S = degs.sum() + 1e-12
        return degs / S
    dd1, dd2 = norm_deg_vec(UG1), norm_deg_vec(UG2)
    # pad to same length
    m = max(len(dd1), len(dd2))
    dd1 = np.pad(dd1, (0, m - len(dd1)))
    dd2 = np.pad(dd2, (0, m - len(dd2)))
    deg_L1 = float(np.sum(np.abs(dd1 - dd2)))

    # Optional approximate GED on binarized graphs
    # ged_val = None
    if approximate_ged:
        try:
            ged_val = nx.graph_edit_distance(UG1, UG2, timeout=ged_timeout)
            ged_val = float(ged_val) if ged_val is not None else np.nan
        except Exception:
            ged_val = np.nan

    out = {
        # vector-level
        **vec,
        **vec_abs,
        "binary_jaccard": bj,

        # spectral / structural
        "spec_L2": spec_L2,
        "spec_Linf": spec_Linf,
        "degree_dist_L1": deg_L1,
        "mcs_approx": mcs_approx_distance(UG1, UG2),
        "laplacian_spectral_dist": laplacian_spectral_distance(UG1, UG2, k=50, normalized=True),
        "netlsd_dist": netlsd_distance(UG1, UG2, num_scales=250),
    }
    if ged_val is not None:
        out["ged_approx"] = ged_val
    return out

def _to_float(x): return float(x.item()) if hasattr(x, "item") else float(x)
def _ek(e, include_qkv=True):
    return (e.parent.name, e.child.name, getattr(e, "qkv", None) if include_qkv else None)

def edge_vector(graph, include_qkv=True):
    items = []
    for e in graph.edges.values():
        key = (e.parent.name, e.child.name, getattr(e, "qkv", None) if include_qkv else None)
        val = float(e.score.item())
        items.append((key, val))
    items.sort(key=lambda t: t[0])          # canonical key order
    keys = [k for k,_ in items]
    vec  = np.array([v for _,v in items], dtype=float)
    return keys, vec

def _align_union(d1, d2):
    keys = sorted(set(d1) | set(d2))
    x = np.array([d1.get(k, 0.0) for k in keys], float)
    y = np.array([d2.get(k, 0.0) for k in keys], float)
    w = np.maximum(np.abs(x), np.abs(y))  # “importance” for trimming/weights
    return x, y, w

def _keep_top_mass(w, mass=0.95, min_k=100):
    if w.size == 0: return np.array([], bool)
    order = np.argsort(-w)
    cs = np.cumsum(w[order])
    k = max(min_k, int(np.searchsorted(cs, mass * (cs[-1] + 1e-12)) + 1))
    keep_idx = order[:min(k, w.size)]
    mask = np.zeros_like(w, dtype=bool); mask[keep_idx] = True
    return mask

def _weighted_pearson(a, b, w):
    w = np.asarray(w, float)
    mu_a = np.average(a, weights=w); mu_b = np.average(b, weights=w)
    ca = a - mu_a; cb = b - mu_b
    num = np.average(ca * cb, weights=w)
    den = (np.sqrt(np.average(ca**2, weights=w)) * np.sqrt(np.average(cb**2, weights=w)) + 1e-12)
    return float(num / den)

def _average_ranks(v):
    order = np.argsort(v)
    ranks = np.empty_like(v, dtype=float); ranks[order] = np.arange(len(v), dtype=float)
    # average ranks for ties
    vals, start = np.unique(v[order], return_index=True)
    for i, s in enumerate(start):
        e = start[i+1] if i+1 < len(start) else len(v)
        ranks[order[s:e]] = (s + e - 1) / 2.0
    return ranks

def get_robust_graph_similarity(id_graph, ood_graph,
                            include_qkv=True,
                            top_mass=0.55, min_k=100, use_abs=True):
    keys1, x = edge_vector(id_graph, include_qkv=include_qkv)
    keys2, y = edge_vector(ood_graph, include_qkv=include_qkv)
    assert keys1 == keys2, "Edge sets/order differ between graphs."
    w = np.maximum(np.abs(x), np.abs(y))
    if use_abs: x, y = np.abs(x), np.abs(y)

    # 1) trim to top mass to kill tail/ties
    mask = _keep_top_mass(w, mass=top_mass, min_k=min_k)
    if mask.sum() < 2:
        return {"srcc_w": np.nan, "tau_b": np.nan, "rbo": np.nan, "sign_agree_w": np.nan}

    x, y, w = x[mask], y[mask], w[mask]

    # 2) weighted Spearman (weight the ranks by magnitude)
    rx, ry = _average_ranks(x), _average_ranks(y)
    srcc_w = _weighted_pearson(rx, ry, w)
    srcc = spearmanr(x, y, nan_policy='omit').correlation

    # 3) Kendall tau-b (tie-adjusted, unweighted) on trimmed set
    tau_b = float(kendalltau(x, y, nan_policy='omit').correlation)

    # 4) rank-biased overlap (RBO) for top-heavy agreement
    def rbo(list_scores_a, list_scores_b, p=0.98):
        A = np.argsort(-list_scores_a)  # ranks by magnitude
        B = np.argsort(-list_scores_b)
        seenA, seenB = set(), set()
        overlap = 0.0; rbo_val = 0.0; wgt = 1.0 - p
        for d in range(1, len(A)+1):
            seenA.add(A[d-1]); seenB.add(B[d-1])
            overlap += len(seenA & seenB) / d
            rbo_val += wgt * (p ** (d-1)) * overlap
        return float(rbo_val)
    rbo_val = rbo(np.abs(x), np.abs(y))

    # 5) sign agreement (weighted)
    same_sign = (np.sign(x) == np.sign(y)).astype(float)
    sign_agree_w = float((w * same_sign).sum() / (w.sum() + 1e-12))

    return {"trim_srcc_w": srcc_w, "trim_srcc": srcc, "trim_tau_b": tau_b, "trim_rbo": rbo_val, "trim_sign_agree_w": sign_agree_w}

def get_attention_mlp_ratio(graph):
    attn_sum = 0.0
    mlp_sum = 0.0
    for edge in graph.edges.values():
        # if not edge.in_graph:
        #     continue
        if isinstance(edge.child, AttentionNode):
            attn_sum += edge.score
        elif isinstance(edge.child, MLPNode):
            mlp_sum += edge.score
    total = attn_sum + mlp_sum
    return (mlp_sum / total).item() if total > 0 else float('nan')

def get_layerwise_score_variance(graph):
    layer_map = graph.build_layer_map()
    layer_scores = {}
    for edge in graph.edges.values():
        layer = layer_map.get(edge.parent.name, None)
        if layer is not None:
            layer_scores[layer] = layer_scores.get(layer, 0.0) + edge.score
    if not layer_scores:
        return float('nan')
    scores = torch.tensor(list(layer_scores.values()), dtype=torch.float32)
    return torch.var(scores, unbiased=False).item()


def get_layerwise_score_entropy(graph, eps=1e-8):
    layer_map = graph.build_layer_map()
    layer_scores = {}

    # Aggregate EAP scores per layer
    for edge in graph.edges.values():
        layer = layer_map.get(edge.parent.name, None)
        if layer is not None:
            layer_scores[layer] = layer_scores.get(layer, 0.0) + abs(edge.score)

    if not layer_scores:
        return float('nan')

    scores = torch.tensor(list(layer_scores.values()), dtype=torch.float32)

    # Normalize to form a probability distribution
    probs = scores / (scores.sum() + eps)

    # Compute entropy (base e)
    entropy = -torch.sum(probs * torch.log(probs + eps)).item()

    return entropy

def compute_edge_norm(graph, *, p=3, in_graph_only=True, per_sign=False):
    """
    Compute ||w||_p over edge scores (residual circuits allowed).
      - p can be 1, 2, float('inf'), etc.
      - If per_sign=True, also return norms for positive and negative edges separately.
      - Uses absolute values for the norm as usual.

    Returns:
      {"L_p": float, "L0": int, "Linf": float (if p==inf), ...}
      If per_sign=True, also includes "pos" and "neg" dicts with the same fields.
    """
    pos, neg = [], []

    for e in graph.edges.values():
        if in_graph_only and not getattr(e, "in_graph", True):
            continue
        # robust float extraction
        s = e.score
        s = float(s.item()) if hasattr(s, "item") else float(s)
        if s > 0:
            pos.append(abs(s))
        elif s < 0:
            neg.append(abs(s))
        # (s == 0) contributes to L0 counting but not magnitude

    def _norm_stats(vals):
        if not vals:
            return {"L_p": 0.0, "L1": 0.0, "L2": 0.0, "Linf": 0.0, "L0": 0}
        t = torch.tensor(vals, dtype=torch.float32)
        # generic p-norm
        Lp = torch.linalg.vector_norm(t, ord=p).item() if math.isfinite(p) else t.abs().max().item()
        return {
            "L_p": Lp,
            "L1": t.abs().sum().item(),
            "L2": torch.linalg.vector_norm(t, ord=2).item(),
            "Linf": t.abs().max().item(),
            "L0": int((t != 0).sum().item()),
        }

    all_vals = pos + neg
    out = _norm_stats(all_vals)
    if per_sign:
        out["pos"] = _norm_stats(pos)
        out["neg"] = _norm_stats(neg)
    return out

def get_weighted_path_depth(graph):
    layer_map = graph.build_layer_map()
    edge_depths, edge_scores = [], []
    for edge in graph.edges.values():
        if edge.in_graph:
            parent_layer = layer_map.get(edge.parent.name, 0)
            child_layer = layer_map.get(edge.child.name, 0)
            depth = max(child_layer - parent_layer, 0)
            edge_depths.append(depth)
            edge_scores.append(edge.score)
    if not edge_scores:
        return float('nan')
    edge_depths = torch.tensor(edge_depths, dtype=torch.float32)
    edge_scores = torch.tensor(edge_scores, dtype=torch.float32)
    return (edge_scores * edge_depths).sum().div(edge_scores.sum()).item()

METRIC_FUNCS = {
    "num_edges": get_num_edges,
    "DDB_deep": get_DDB_deep,
    "DDB_global": get_DDB_global,
    "DDB_out": get_DDB_out,
    "generalization_graph_metrics": get_generalization_graph_metrics,
    "CSS": get_distance_from_id,
}

# ---------------------------------------
# Cache wrapper
# ---------------------------------------
CACHE_DIR = f"metrics/{task}_{split}{circuit_type}"
os.makedirs(CACHE_DIR, exist_ok=True)

def cached_metric(model_id: int, metric: str):
    ck = f"{CACHE_DIR}/{metric}_{model_id}.p"
    if os.path.exists(ck):
        with open(ck, "rb") as f:
            return pickle.load(f)
    g, per_example_scores, ob_auc, bg_auc = load_graph(model_id)
    if metric == "circuit_instability":
        val = compute_circuit_instability(g, per_example_scores)
    elif metric == "normed_circuit_instability":
        val = compute_normed_circuit_instability(g, per_example_scores)
    elif metric == "bg_object_scores":
        val = get_bg_object_scores(g, ob_auc, bg_auc)
    elif metric == "distance_from_id":
        val = get_distance_from_id(g)
    else:
        val = METRIC_FUNCS[metric](g)
    with open(ck, "wb") as f:
        pickle.dump(val, f)
    return val

def compute_metric_for_sweep(task, split, circuit_type, metrics, probit=False):
    SWEEP_CSV = f"/home/yxpengcs/PycharmProjects/vit-spurious-robustness/output/{task}_sweep_results_new.csv"
    OUT_CSV = f"metrics/{task}_model_{split}{circuit_type}_data.csv"
    if 'waterbirds' in task:
        sweep = pd.read_csv(SWEEP_CSV)[
            ["model_id", "linear_probe", "weight_decay", 'model_type', 'use_adam', 'learning_rate', "val_id_acc",
             "test_acc_0", "test_acc_1", "test_acc_2", "test_acc_3"]]
        group_sizes = np.array([2255, 2255, 642, 642])
        acc = np.stack([sweep['test_acc_0'].tolist(), sweep['test_acc_1'].tolist(), sweep['test_acc_2'].tolist(),
                        sweep['test_acc_3'].tolist()], axis=1)
        sweep['test_id_acc'] = (acc * group_sizes).sum(axis=1) / group_sizes.sum()
    elif 'PACS-photo' in task:
        sweep = pd.read_csv(SWEEP_CSV)[
            ["model_id", "linear_probe", "weight_decay", 'model_type', 'use_adam', 'learning_rate', "val_id_acc", "val_id_f1",
             "test_acc_art_painting", "test_acc_cartoon", "test_acc_sketch",
             "train_acc", "train_f1", "val_AC", "val_ANE", "val_l2", "val_EMD", "test_AC_art_painting",
             "test_ANE_art_painting", "test_l2_art_painting", "test_EMD_art_painting", "test_AC_cartoon",
             "test_ANE_cartoon", "test_l2_cartoon", "test_EMD_cartoon",
             "test_AC_sketch", "test_ANE_sketch", "test_l2_sketch", "test_EMD_sketch"
             ]]
    elif task == 'PACS':
        sweep = pd.read_csv(SWEEP_CSV)[
            ["model_id", "linear_probe", "weight_decay", 'model_type', 'use_adam', 'learning_rate', "val_id_acc", "val_id_f1",
             "test_acc_art_painting", "test_acc_cartoon", "test_acc_photo",
             "train_acc", "train_f1", "val_AC", "val_ANE", "val_l2", "val_EMD", "test_AC_art_painting",
             "test_ANE_art_painting", "test_l2_art_painting", "test_EMD_art_painting", "test_AC_cartoon",
             "test_ANE_cartoon", "test_l2_cartoon", "test_EMD_cartoon",
             "test_AC_photo", "test_ANE_photo", "test_l2_photo", "test_EMD_photo"
             ]]
    elif 'camelyon17' in task:
        sweep = pd.read_csv(SWEEP_CSV)[
            ["model_id", "linear_probe", "weight_decay", 'model_type', 'use_adam', 'learning_rate', "val_id_acc", "val_id_f1",
             "test_acc_hospital1", "test_acc_hospital2",
             "val_AC", "val_ANE", "val_l2", "val_EMD", "test_AC_hospital1",
             "test_ANE_hospital1", "test_l2_hospital1", "test_EMD_hospital1", "test_AC_hospital2", "test_ANE_hospital2", "test_l2_hospital2", "test_EMD_hospital2",]]
    elif 'terra-incognita-38' in task:
        sweep = pd.read_csv(SWEEP_CSV)[
            ["model_id", "linear_probe", "weight_decay", 'model_type', 'use_adam', 'learning_rate', "val_id_acc", "val_id_f1",
             "test_acc_location_43", "test_f1_location_43", "test_acc_location_46", "test_f1_location_46", "test_acc_location_100", "test_f1_location_100",
             "train_acc", "train_f1", "val_AC", "val_ANE", "val_l2", "val_EMD", "test_AC_location_46",
             "test_ANE_location_46", "test_l2_location_46", "test_EMD_location_46", "test_AC_location_100", "test_ANE_location_100", "test_l2_location_100", "test_EMD_location_100",
             "test_AC_location_43","test_ANE_location_43","test_l2_location_43","test_EMD_location_43"]]
    else:
        sweep = pd.read_csv(SWEEP_CSV)
    ckpt_root_path = '/home/yxpengcs/PycharmProjects/vit-spurious-robustness'
    # num_components = [100 for i in range(len(df))]
    # id_accs = df["val_id_acc"].tolist()
    # generalize_accs = df["test_acc_2"].tolist()
    model_ids = sweep["model_id"].tolist()
    if probit:
        eps = 1e-6
        sweep["val_id_acc"] = norm.ppf(np.clip(sweep.val_id_acc, eps, 1 - eps))
        sweep["test_id_acc"] = norm.ppf(np.clip(sweep.test_acc_2, eps, 1 - eps))
    if os.path.exists(OUT_CSV):
        exist = True
        base_df = pd.read_csv(OUT_CSV)
        if "model_id" not in base_df.columns:
            base_df["model_id"] = model_ids
    else:
        exist = False
        base_df = pd.DataFrame({"model_id": sweep["model_id"]})

    for metric in metrics:
        if metric not in METRIC_FUNCS:
            raise ValueError(f"Unknown metric: {metric}")
        if metric in base_df.columns:
            print(f"Skipping {metric}, already in {OUT_CSV}")
            continue

        print(f"Computing {metric}…")

        rows = []
        for mid in tqdm(sweep.model_id.tolist()):
            output = cached_metric(mid, metric)
            row = flatten_metric_output(metric, output)
            row["model_id"] = mid  # Add model_id for later merge
            rows.append(row)

        # Combine all rows into a dataframe
        metric_df = pd.DataFrame(rows)

        # Merge on model_id
        base_df = pd.merge(base_df, metric_df, on="model_id", how="left")
    rename_map = {}
    drop_cols = []
    for col in base_df.columns:
        if col.endswith("_x"):
            base = col[:-2]  # strip "_x"
            rename_map[col] = base
            if f"{base}_y" in base_df.columns:
                drop_cols.append(f"{base}_y")
    base_df = base_df.drop(columns=drop_cols)
    base_df.to_csv(OUT_CSV, index=False)
    # merged = sweep.merge(base_df, on="model_id", how="left") if not exist else base_df
    # merged.to_csv(OUT_CSV, index=False)
    print(f"Saved metrics to {OUT_CSV}")

def compute_metric_for_ood_predict(task, split, circuit_type, metrics,
            probit=False, model_id=0, topn=200, residual=False):
    resid_str = "_resid" if residual else ""
    SWEEP_CSV = f"/home/yxpengcs/PycharmProjects/vit-spurious-robustness/output/{task}_sweep_results_new.csv"
    OUT_CSV = f"metrics/{task}_model_{model_id}{circuit_type}{resid_str}_{topn}_data.csv"

    sweep = pd.read_csv(SWEEP_CSV)
    if probit:
        eps = 1e-6
        sweep["val_id_acc"] = norm.ppf(np.clip(sweep.val_id_acc, eps, 1 - eps))
        sweep["test_id_acc"] = norm.ppf(np.clip(sweep.test_acc_2, eps, 1 - eps))

    test_cols = sweep.filter(regex=r'^test_acc_').columns.tolist()
    domains = [c.replace("test_acc_", "") for c in test_cols]
    exclude_domains = ['0', '1', '2', '3', 'photo', 'cartoon', 'art_painting', 'id2']
    for exclude_domain in exclude_domains:
        if exclude_domain in domains:
            domains.remove(exclude_domain)
    domains += ["id"]

    # ---- base table: ONE ROW PER DOMAIN ----
    if os.path.exists(OUT_CSV):
        base_df = pd.read_csv(OUT_CSV)
        if "domain" not in base_df.columns:
            base_df = pd.DataFrame({"domain": domains})
        else:
            base_df = base_df[base_df["domain"].isin(domains)].copy()
    else:
        base_df = None

    rows = []
    recompute_existing = []

    # --- figure out which metrics are missing ---
    if base_df is not None:
        existing_metrics = set(base_df.columns) - {"domain"}
        missing_metrics = [m for m in metrics if m not in existing_metrics]
        if missing_metrics:
            print(f"[INFO] Will recompute metrics for existing domains: {missing_metrics}")
            recompute_existing = missing_metrics

    for domain in tqdm(domains):
        if base_df is not None and domain in base_df["domain"].tolist() and not recompute_existing:
            continue

        gpath = (
            f"circuits/EAP-IG-inputs_mean-positional_edge_train_kl_divergence/"
            f"{task}-mean-{domain.replace('_', '-')}_sweep_{model_id}/importances.pt"
        )
        id_gpath = (
            f"circuits/EAP-IG-inputs_mean-positional_edge_train_kl_divergence/"
            f"{task}-mean-id_sweep_{model_id}/importances.pt"
        )
        if residual:
            gpath = (
                f"circuits/EAP-IG-inputs_mean-positional_edge_train_kl_divergence/"
                f"{task}-mean-{domain.replace('_', '-')}_sweep_{model_id}/residual_importances.pt"
            )

        g = Graph.from_pt(gpath)
        id_g = Graph.from_pt(id_gpath)
        if residual:
            g.apply_topn(TOPN, True, level="edge", prune=True)
        else:
            g.apply_topn(TOPN, False, level="edge", prune=True)
            id_g.apply_topn(TOPN, False, level="edge", prune=True)

        row = {"domain": domain}

        # Case 1: completely new domain → compute all metrics
        if base_df is None or domain not in base_df["domain"].tolist():
            for metric in metrics:
                print(f"[{domain}] computing metric {metric}")
                if metric == "circuit_instability":
                    val = compute_circuit_instability(g, per_example_scores)
                elif metric == "normed_circuit_instability":
                    val = compute_normed_circuit_instability(g, per_example_scores)
                elif metric == "distance_from_id":
                    val = get_distance_from_id(id_g, g)
                elif metric == "layer_distance_from_id":
                    val = get_layer_distance_from_id(id_g, g)
                elif metric == "robust_graph_similarity":
                    val = get_robust_graph_similarity(id_g, g)
                else:
                    val = METRIC_FUNCS[metric](g)
                val = flatten_metric_output(metric, val)
                row.update(val)
            rows.append(row)

        # Case 2: domain exists but missing some metrics
        elif recompute_existing:
            for metric in recompute_existing:
                print(f"[{domain}] recomputing missing metric {metric}")
                if metric == "distance_from_id":
                    val = get_distance_from_id(id_g, g)
                else:
                    val = METRIC_FUNCS[metric](g)
                val = flatten_metric_output(metric, val)
                row.update(val)
            row["domain"] = domain
            rows.append(row)

    if not rows:
        print("[INFO] No rows produced; check domains/paths/metrics.")
        return

    out_df = pd.DataFrame(rows)

    if base_df is not None:
        out_df = pd.merge(base_df, out_df, on="domain", how="outer")
    out_df.to_csv(OUT_CSV, index=False)

    print(f"[OK] Saved metrics to {OUT_CSV} with {len(out_df)} domains and {out_df.shape[1] - 1} metric columns.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute circuit metrics for sweep or OOD prediction.")
    parser.add_argument("--mode", type=str, choices=["sweep", "ood_predict"], required=True,
                        help="Computation mode: 'sweep' or 'ood_predict'")
    parser.add_argument("--task", type=str, default="PACS-photo", help="Dataset/task name, e.g. 'PACS-photo'")
    parser.add_argument("--split", type=str, default="sketch", help="Split or domain name, e.g. 'sketch'")
    parser.add_argument("--metrics", type=str, nargs="+", required=True, help="List of metrics to compute")
    parser.add_argument("--probit", action="store_true", help="Apply probit transform to accuracies")
    parser.add_argument("--circuit_type", type=str, default="_kl", help="Circuit type suffix")
    parser.add_argument("--model_id", type=int, default=0, help="Model ID (for OOD mode)")
    parser.add_argument("--topn", type=int, default=200, help="Top-N pruning level")
    parser.add_argument("--residual", action="store_true", help="Use residual importances instead of standard")

    args = parser.parse_args()

    if args.mode == "sweep":
        compute_metric_for_sweep(args.task, args.split, args.circuit_type, args.metrics, probit=args.probit)
    else:
        compute_metric_for_ood_predict(
            args.task, args.split, args.circuit_type, args.metrics,
            probit=args.probit, model_id=args.model_id, topn=args.topn, residual=args.residual
        )