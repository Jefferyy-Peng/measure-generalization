from eap.graph import Graph
import os

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--graph_paths", type=str, nargs='+', required=True)
    parser.add_argument("--num_components", type=int, nargs='+', required=True)
    parser.add_argument("--resid_str", type=str, default='')

    args = parser.parse_args()

    abs = True

    for idx, graph_path in enumerate(args.graph_paths):
        if graph_path.endswith('json'):
            g = Graph.from_json(graph_path)
        elif graph_path.endswith('pt'):
            g = Graph.from_pt(graph_path)
        g.apply_topn(args.num_components[idx], abs, level='edge', prune=False)
        # g.prune()
        g.to_graphviz(os.path.join(os.path.dirname(graph_path), f'n-{args.num_components[idx]}-abs{abs}{args.resid_str}.png'))