from pathlib import Path
import pandas as pd

def crossval_result_csv_path(prediction_csv_dir, result_prefix="HIPIT-Affinity"):
    return Path(prediction_csv_dir) / f"{result_prefix}.csv"

def dataset_result_csv_path(prediction_csv_dir, dataset_name, result_prefix="HIPIT-Affinity"):
    return Path(prediction_csv_dir) / f"{result_prefix}_{dataset_name}.csv"

def _fold_summary_values(fold_result):
    from Training import _available_groups, _group_metrics
    values = {
        "TrainSize": fold_result["train_size"],
        "ValSize": fold_result["val_size"],
        "BestEpoch": fold_result["best_epoch"],
        "BestValRMSE": fold_result["best_val_rmse"],
        "BestValMAE": fold_result["best_val_mae"],
        "BestValR": fold_result["best_val_r"],
        "BestValR2": fold_result["best_val_r2"],
    }
    groups = _available_groups(fold_result.get("test_outputs", {}))
    if groups:
        for group_name, members in groups:
            metrics, _ = _group_metrics(fold_result, members)
            for metric_name in ("rmse", "mae", "r", "r2"):
                values[f"{group_name}_{metric_name.upper()}"] = metrics[metric_name]
    else:
        for dataset_name, metrics in fold_result["test_metrics"].items():
            for metric_name, metric_value in metrics.items():
                values[f"{dataset_name}_{metric_name.upper()}"] = metric_value
    return values

def save_crossval_result_csv(prediction_csv_dir, fold_results, result_prefix="HIPIT-Affinity", include_validation=False):
    rows = []
    rows_by_dataset = {}
    for fold_result in fold_results:
        fold_idx = fold_result["fold"]
        fold_summary = _fold_summary_values(fold_result)
        outputs_by_name = dict(fold_result["test_outputs"])
        if include_validation:
            outputs_by_name = {"Validation": fold_result["val_outputs"], **outputs_by_name}
        for dataset_name, outputs in outputs_by_name.items():
            for pdb_id, label, pred in zip(outputs["ids"], outputs["labels"], outputs["preds"]):
                row = {
                    "Fold": fold_idx,
                    "Dataset": dataset_name,
                    "PDB": pdb_id,
                    "Actual": label,
                    "Predicted": pred,
                }
                row.update(fold_summary)
                rows.append(row)
                rows_by_dataset.setdefault(dataset_name, []).append(row)
    csv_path = crossval_result_csv_path(prediction_csv_dir, result_prefix=result_prefix)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    for dataset_name, dataset_rows in rows_by_dataset.items():
        dataset_csv_path = dataset_result_csv_path(prediction_csv_dir, dataset_name, result_prefix=result_prefix)
        dataset_csv_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(dataset_rows).to_csv(dataset_csv_path, index=False)
