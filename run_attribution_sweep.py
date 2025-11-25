import argparse
import os
import pickle
from functools import partial

from dataset import VisionEAPDataset
from eap.graph import Graph
from eap.attribute import attribute
from eap.attribute_node import attribute_node

from dataset_utils import setup_dataset
from metrics import get_metric

from model_utils import get_model, get_model_from_ckpt
from print_results import COL_MAPPING

TASKS_TO_HF_NAMES = {
    'ioi': 'ioi',
    'mcqa': 'copycolors_mcqa',
    'arithmetic_addition': 'arithmetic_addition',
    'arithmetic_subtraction': 'arithmetic_subtraction',
    'arc_easy': 'arc_easy',
    'arc_challenge': 'arc_challenge',
}

MODEL_NAME_TO_FULLNAME = {
    "gpt2": "gpt2-small",
    "qwen2.5": "Qwen/Qwen2.5-0.5B",
    "gemma2": "google/gemma-2-2b",
    "llama3": "meta-llama/Llama-3.1-8B"
}

if __name__ == "__main__":
    csv_path = "/path/to/your/models"
    import pandas as pd

    get_perexample_scores = False
    circuit_dir = 'circuits'
    fragment = None
    batch_size = 30
    metric_name = 'kl_divergence'
    method = 'EAP-IG-inputs'
    ablation = 'mean-positional'
    ig_steps = 5
    optimal_ablation_path = None
    device = 'cuda:6'
    level = 'edge'
    df = pd.read_csv(csv_path)
    tasks = ['camelyon17-mean-hospital2']
    ckpt_root_path = 'root/path/to/models'
    split = 'train'

    for row in df.itertuples(index=False):
        model_id = row.model_id
        # if model_id != 0:
        #     continue
        model_name = row.model_type
        model_ckpt_path = row.checkpoint
        for task in tasks:
            method_name_saveable = f"{method}_{ablation}_{level}_{split}_{metric_name}"
            circuit_path = os.path.join(circuit_dir, method_name_saveable, f"{task.replace('_', '-')}_sweep_{model_id}")
            # if os.path.exists(circuit_path):
            #     continue
            # else:
            #     os.makedirs(circuit_path, exist_ok=True)

            model = get_model_from_ckpt(model_name, task, model_ckpt_path, ckpt_root_path, device=device)
            model.cfg.use_split_qkv_input = True
            model.cfg.use_attn_result = True
            model.cfg.use_hook_mlp_in = True
            model.cfg.ungroup_grouped_query_attention = True
            graph = Graph.from_model(model)
            dataset, intervention_dataset = setup_dataset(task, split=split, model_name=model_name, num_examples=1000000000, fragment=fragment, device=device)

            dataloader = dataset.to_dataloader(batch_size=batch_size)
            intervention_dataloader = intervention_dataset.to_dataloader(batch_size=batch_size)
            metric = get_metric(metric_name, task, model, model)
            attribution_metric = partial(metric, mean=True, loss=True)
            if level == 'edge':
                perexample_scores = attribute(model, graph, dataloader, attribution_metric, method, ablation, get_perexample_scores=get_perexample_scores, intervention_dataloader=intervention_dataloader,
                          ig_steps=ig_steps, optimal_ablation_path=optimal_ablation_path, device=device, task=task, model_id=model_id)
            else:
                attribute_node(model, graph, dataloader, attribution_metric, method,
                               ablation, neuron=level == 'neuron', ig_steps=ig_steps,
                               optimal_ablation_path=optimal_ablation_path)

            # Save the graph
            fragment_number = '_' + str(fragment) if fragment is not None else ''
            graph.to_pt(f'{circuit_path}/importances{fragment_number}.pt')
            if perexample_scores:
                with open(f'{circuit_path}/perexample_importances{fragment_number}.p', 'wb') as file:
                    pickle.dump(perexample_scores, file)
