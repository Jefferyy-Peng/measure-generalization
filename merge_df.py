import pandas as pd
import numpy as np

def attach_domain_test_metrics(out_df: pd.DataFrame,
                               sweep: pd.DataFrame,
                               model_id: int | None = None,
                               prefix: str = "test_",
                               add_prefix_to_cols: bool = False) -> pd.DataFrame:
    """
    From a sweep df with columns like:
      test_acc_motion_blur-3, test_AC_motion_blur-3, test_ANE_motion_blur-3, test_l2_motion_blur-3, test_EMD_motion_blur-3
    produce a table keyed by (model_id, domain) with columns [acc, AC, ANE, l2, EMD] (or prefixed),
    and left-merge it into out_df.

    out_df must have ['domain'] and (optionally) ['model_id'].
    If out_df doesn't have model_id and you pass model_id, this function will add it.
    """

    # 1) collect test_* columns
    test_cols = [c for c in sweep.columns if c.startswith(prefix)]
    if not test_cols:
        return out_df

    # 2) long form: (model_id, key, value)
    need_keys = ['model_id'] if 'model_id' in sweep.columns else []
    long = sweep[need_keys + test_cols].melt(id_vars=need_keys,
                                             var_name="key",
                                             value_name="value")

    # 3) split key -> metric, domain
    # pattern: test_<metric>_<domain>  (domain can contain underscores/hyphens)
    md = long['key'].str.extract(r'^' + prefix + r'(?P<metric>[^_]+)_(?P<domain>.+)$')
    long = pd.concat([long.drop(columns=['key']), md], axis=1).dropna(subset=['metric', 'domain'])

    # ensure numeric
    long['value'] = pd.to_numeric(long['value'], errors='coerce')

    # 4) filter to one model_id if requested
    if model_id is not None and 'model_id' in long.columns:
        long = long[long['model_id'] == model_id].copy()

    # 5) pivot to wide: (model_id, domain) -> metrics
    index_cols = (['model_id', 'domain'] if 'model_id' in long.columns else ['domain'])
    wide = long.pivot_table(index=index_cols, columns='metric', values='value', aggfunc='first').reset_index()

    # optional: add "test_" prefix back to metric columns for clarity
    if add_prefix_to_cols:
        rename_map = {c: f"{prefix}{c}" for c in wide.columns if c not in index_cols}
        wide = wide.rename(columns=rename_map)

    # 6) make sure out_df has model_id if needed
    if 'model_id' in wide.columns and 'model_id' not in out_df.columns and model_id is not None:
        out_df = out_df.copy()
        out_df['model_id'] = model_id

    # 7) merge
    on_keys = ['domain'] + (['model_id'] if 'model_id' in wide.columns and 'model_id' in out_df.columns else [])
    for df in (out_df, wide):
        df['domain'] = df['domain'].astype(str).str.strip()
    merged = out_df.merge(wide, on=on_keys, how='left', suffixes=('', '_y'))
    for col in wide.columns:
        if col in on_keys:  # skip join keys
            continue
        if col in merged.columns and f"{col}_y" in merged.columns:
            merged[col] = merged[col].fillna(merged[f"{col}_y"])
            merged = merged.drop(columns=[f"{col}_y"])

    # ---- 8) attach validation metrics for domain == 'id' ----
    val_cols = [c for c in ['val_id_acc','val_id_f1','val_AC','val_ANE','val_EMD']
                if c in sweep.columns]
    if val_cols:
        # make sure columns exist
        for c in val_cols:
            if c not in merged.columns:
                merged[c] = np.nan

        # pick the source values from sweep (for this model_id if given)
        if 'model_id' in sweep.columns and model_id is not None:
            src = sweep.loc[sweep['model_id'] == model_id, val_cols].head(1)
        else:
            # if there’s only one unique set of val stats, broadcast it
            src = sweep[val_cols].drop_duplicates().head(1)

        if not src.empty:
            mask_id = merged['domain'].astype(str).str.strip().str.lower().eq('id')
            merged.loc[mask_id, val_cols] = src.iloc[0].values

        val_to_main = {
            "acc": "val_id_acc",
            "f1": "val_id_f1",
            "AC": "val_AC",
            "ANE": "val_ANE",
            "EMD": "val_EMD",
        }

        # mask for the 'id' row
        mask_id = merged['domain'].astype(str).str.strip().str.lower().eq('id')

        for dst, src in val_to_main.items():
            if src in merged.columns:
                if dst not in merged.columns:
                    merged[dst] = np.nan
                # copy validation value(s) into the id row
                merged.loc[mask_id, dst] = merged.loc[mask_id, src].values

        # drop the val_* columns after merging (optional)
        merged.drop(columns=[c for c in val_to_main.values() if c in merged.columns],
                    inplace=True, errors='ignore')

    return merged

model_id = 0
SWEEP_CSV = f"/home/yxpengcs/PycharmProjects/vit-spurious-robustness/output/camelyon17-set2_sweep_results_new.csv"
OUT_CSV = f"metrics/camelyon17-set2_model_{model_id}_kl_200_data_new.csv"
base_df = pd.read_csv(OUT_CSV)
sweep = pd.read_csv(SWEEP_CSV)

out_df = attach_domain_test_metrics(base_df, sweep, model_id=model_id, add_prefix_to_cols=False)

out_df.to_csv(OUT_CSV)