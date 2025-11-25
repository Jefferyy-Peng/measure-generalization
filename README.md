# Circuit Metric Pipeline

This repository provides a complete pipeline for **discovering**, **visualizing**, and **quantitatively analyzing** Vision Transformer (ViT) circuits using the EAP/IG framework.

---

## 1. Circuit Discovery

Run `run_attribution_sweep.py` to extract **EAP/IG attribution graphs** for your trained models.

### Steps:
1. Open `run_attribution_sweep.py`.
2. Modify the `csv_path` variable to point to the CSV file containing your model information.
   ```python
   csv_path = "/path/to/your_model_sweep.csv"

3.  Update the tasks list to include the generalization tasks you wish to analyze.
    Each task should be named using the format:

<!-- end list -->

```php-template
"<source domain>-mean-<target domain>"
```

For example:

```python
tasks = [
    "photo-mean-sketch",
    "photo-mean-cartoon",
    "photo-mean-art_painting"
]
```

4.  Run the script:

<!-- end list -->

```bash
python run_attribution_sweep.py
```

This step will generate circuit importances (`importances.pt`) for each model-domain pair under:

```swift
circuits/EAP-IG-inputs_mean-positional_edge_train_kl_divergence/
```

## 2\. Circuit Visualization

After extraction, visualize the top-k important circuits.

Run:

```bash
bash run_top.sh
```

This will generate and save visualizations of the extracted circuits for inspection.
You can adjust the number of top edges visualized (default `TOPN=200`) and the visualization style inside `run_top.sh`.

## 3\. Circuit Metric Computation

Once circuits are extracted, compute quantitative circuit metrics (e.g., connectivity, depth bias, spectral metrics, similarity measures) by running:

```bash
python compute_metrics.py \
    --mode sweep \
    --task PACS-photo \
    --split sketch \
    --metrics DDB_global DDB_deep DDB_out 
```

### Arguments

| Argument | Description |
| :--- | :--- |
| `--mode` | Choose the computation stage: \<ul\>\<li\>`sweep`: Pre-deployment metrics across trained models.\</li\>\<li\>`ood_predict`: Post-deployment metrics comparing ID vs. OOD circuits.\</li\>\</ul\> |
| `--task` | The dataset or task name (e.g., `PACS-photo`). |
| `--split` | Target domain for evaluation (e.g., `sketch`). |
| `--metrics` | List of metrics to compute. Multiple metrics can be specified. |
| `--probit` | Optional. Apply probit transform to accuracies for correlation analysis. |

### Examples

**Pre-deployment (Sweep) mode:**

```bash
python compute_metrics.py \
    --mode sweep \
    --task PACS-photo \
    --split sketch \
    --metrics CSS
```

**Post-deployment (OOD Predict) mode:**

```bash
python compute_metrics.py \
    --mode ood_predict \
    --task PACS-photo \
    --split sketch \
    --metrics distance_from_id robust_graph_similarity \
    --model_id 0
```

### Output

  * All computed metrics are saved in:

<!-- end list -->

```php-template
metrics/<task>_<split>_<metric_type>.csv
```

  * Circuit visualizations are saved in:

<!-- end list -->

```php-template
figures/<task>_<split>/
```

  * Raw circuit data (importances) are saved in:

<!-- end list -->

```swift
circuits/EAP-IG-inputs_mean-positional_edge_train_kl_divergence/
```

## Summary

| Stage | Script | Description |
| :--- | :--- | :--- |
| **Circuit Discovery** | `run_attribution_sweep.py` | Extract circuit importance values for each domain. |
| **Visualization** | `bash run_top.sh` | Visualize the top edges/nodes in the discovered circuits. |
| **Metric Computation** | `compute_metrics.py` | Compute pre- or post-deployment circuit metrics. |

-----

## Notes

  * Ensure consistent task naming between `run_attribution_sweep.py` and `compute_metrics.py`.
  * Metrics can include: `generalization_graph_metrics`, `CSS`, `distance_from_id`, `robust_graph_similarity`, `layer_distance_from_id`, etc.
  * For reproducibility, freeze your environment:

<!-- end list -->

```bash
conda env export > environment.yml
```