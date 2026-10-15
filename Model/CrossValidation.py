from pathlib import Path
import numpy as np
from tqdm import tqdm
from DataLoader import build_fold_dataset, build_test_loaders
from Training import (
    _run_single_fold,
    _log_fold_start,
    _mean_metric,
    _metric_summary_text,
    _available_groups,
    _group_metrics,
)
from Utils import set_seed
from Summary import save_crossval_result_csv

def _fold_model_state_path(base_model_state_path, fold_idx):
    return base_model_state_path.with_name(f"{base_model_state_path.stem}_fold{fold_idx}{base_model_state_path.suffix}")

def _fold_log_dir(base_log_dir, fold_idx):
    return base_log_dir / f"fold_{fold_idx}"

def _log_crossval_average(fold_results):
    if not fold_results:
        return
    fold_label = f"{len(fold_results)}-Fold"
    best_epoch_mean = _mean_metric(fold_results, lambda fold_result: fold_result["best_epoch"])
    val_outputs = {
        "rmse": _mean_metric(fold_results, lambda fold_result: fold_result["best_val_rmse"]),
        "mae": _mean_metric(fold_results, lambda fold_result: fold_result["best_val_mae"]),
        "r": _mean_metric(fold_results, lambda fold_result: fold_result["best_val_r"]),
        "r2": _mean_metric(fold_results, lambda fold_result: fold_result["best_val_r2"]),
    }
    tqdm.write(f"{fold_label} Best epoch={best_epoch_mean:.2f}|{_metric_summary_text('BestVal', val_outputs)}")
    groups = _available_groups(fold_results[0].get("test_outputs", {}))
    if groups:
        for group_name, members in groups:
            per_fold = [_group_metrics(fold_result, members) for fold_result in fold_results]
            group_size = per_fold[0][1]
            group_outputs = {
                metric_name: float(np.mean([metrics[metric_name] for metrics, _ in per_fold]))
                for metric_name in ("rmse", "mae", "r", "r2")
            }
            tqdm.write(f"{fold_label} {_metric_summary_text(f'{group_name:<8}', group_outputs)}   (n={group_size})")
    else:
        for dataset_name in fold_results[0]["test_metrics"].keys():
            dataset_outputs = {
                metric_name: _mean_metric(
                    fold_results,
                    lambda fold_result, name=dataset_name, metric=metric_name: fold_result["test_metrics"][name][metric],
                )
                for metric_name in ("rmse", "mae", "r", "r2")
            }
            tqdm.write(f"{fold_label} {_metric_summary_text(dataset_name, dataset_outputs)}")

def _run_crossval_for_seed(
    train_params,
    training_cache,
    residue_in_dim,
    token_in_dim,
    residue_graph_root,
    cluster_root,
    loader_kwargs,
    eval_loader_kwargs,
):
    set_seed(train_params.FOLD_SEED)
    Path(train_params.LOG_DIR).mkdir(parents=True, exist_ok=True)
    Path(train_params.MODEL_STATE_PATH).parent.mkdir(parents=True, exist_ok=True)
    Path(train_params.PREDICTION_CSV_DIR).mkdir(parents=True, exist_ok=True)

    num_folds = 5
    if not train_params.USE_CROSS_VALIDATION:
        fold_indices = ["fixed"]
    else:
        fold_indices = list(range(1, num_folds + 1))
    base_model_state_path = Path(train_params.MODEL_STATE_PATH)
    base_log_dir = Path(train_params.LOG_DIR)
    fold_results = []

    mode_text = "Crossvalidation Training" if train_params.USE_CROSS_VALIDATION else "Training"
    tqdm.write(f"Seed {train_params.FOLD_SEED}: starting {mode_text}")
    for fold_idx in fold_indices:
        train_dataset = build_fold_dataset(training_cache, train_params, fold_idx, "Train", num_folds)
        val_dataset = build_fold_dataset(training_cache, train_params, fold_idx, "Validation", num_folds)
        test_loaders = build_test_loaders(train_params, eval_loader_kwargs, fold_idx, residue_graph_root, cluster_root)
        if train_params.USE_CROSS_VALIDATION:
            train_params.MODEL_STATE_PATH = str(_fold_model_state_path(base_model_state_path, fold_idx))
            train_params.LOG_DIR = str(_fold_log_dir(base_log_dir, fold_idx))
        else:
            train_params.MODEL_STATE_PATH = str(base_model_state_path)
            train_params.LOG_DIR = str(base_log_dir)
        _log_fold_start(train_params, fold_idx, train_dataset, val_dataset, test_loaders)
        fold_results.append(
            _run_single_fold(
                train_params=train_params,
                fold_idx=fold_idx,
                train_dataset=train_dataset,
                val_dataset=val_dataset,
                residue_in_dim=residue_in_dim,
                token_in_dim=token_in_dim,
                loader_kwargs=loader_kwargs,
                eval_loader_kwargs=eval_loader_kwargs,
                test_loaders=test_loaders,
            )
        )

    train_params.MODEL_STATE_PATH = str(base_model_state_path)
    train_params.LOG_DIR = str(base_log_dir)
    save_crossval_result_csv(
        train_params.PREDICTION_CSV_DIR,
        fold_results,
        result_prefix=train_params.RESULT_PREFIX,
        include_validation=train_params.USE_CROSS_VALIDATION,
    )
    if train_params.USE_CROSS_VALIDATION:
        _log_crossval_average(fold_results)
    return fold_results
