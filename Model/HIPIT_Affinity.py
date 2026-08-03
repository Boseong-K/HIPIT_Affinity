import os
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import argparse
from pathlib import Path
import torch
from CrossValidation import _run_crossval_for_seed
from DataLoader import (
    build_collate_fn,
    build_training_cache,
    load_sample_residue_graph,
    load_sample_token,)
from Utils import build_loader_kwargs, setup_runtime

DATASET_ROOT = ""
REPRESENTATION_ROOT = ""
OUTPUT_ROOT = ""

class TrainParams:
    def __init__(self, test_datasets, output_root=OUTPUT_ROOT, model_name="HIPIT-Affinity", log_dir=None):
        output_root = Path(output_root)
        self.DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.USE_CROSS_VALIDATION = True
        self.BATCH_SIZE = 16
        self.NUM_WORKERS = 8
        self.EPOCHS = 100
        self.USE_EGNN_BRANCH = True
        self.USE_IT_BRANCH = True
        self.RESULT_PREFIX = model_name
        self.MODEL_STATE_PATH = str(output_root / f"{model_name}.pt")
        self.PREDICTION_CSV_DIR = str(output_root / model_name)
        self.LOG_DIR = str(log_dir or (output_root / "runs" / model_name))
        self.TEST_DATASETS = dict(test_datasets or {})

def discover_dataset_names(dataset_root):
    dataset_root = Path(dataset_root)
    if not dataset_root.is_dir():
        return []
    return [sub.name for sub in sorted(dataset_root.iterdir())
            if sub.is_dir() and sub.name != "Train" and (sub / f"{sub.name}.csv").exists()]

def build_test_datasets(dataset_root, representation_root, names, id_col="PDB", value_col="pKD"):
    dataset_root = Path(dataset_root)
    residue_root = Path(representation_root) / "Residue_Graph"
    token_root = Path(representation_root) / "Atom_Token"
    datasets = {}
    for name in names:
        datasets[name] = {
            "RESIDUE_DATA_ROOT": str(residue_root / name),
            "TOKEN_DATA_ROOT": str(token_root / name),
            "LABEL_FILE": str(dataset_root / name / f"{name}.csv"),
            "LABEL_ID_COL": id_col,
            "LABEL_VALUE_COL": value_col,
        }
    return datasets

def parse_args():
    parser = argparse.ArgumentParser(description="Run HIP-IT affinity training (fixed-split or cross-validation).")
    parser.add_argument("--dataset-root", default=str(DATASET_ROOT))
    parser.add_argument("--representation-root", default=str(REPRESENTATION_ROOT))
    parser.add_argument("--output-root", default=str(OUTPUT_ROOT))
    parser.add_argument("--model-name", default="HIPIT-Affinity")
    parser.add_argument("--datasets", nargs="+", default=[])
    return parser.parse_args()

def train_model(dataset_root, representation_root, output_root, model_name, datasets):
    dataset_root = Path(dataset_root)
    representation_root = Path(representation_root)
    residue_graph_root = representation_root / "Residue_Graph"
    cluster_root = representation_root / "Interface_Cluster"
    residue_train_root = residue_graph_root / "Train"
    token_train_root = representation_root / "Atom_Token" / "Train"
    label_file = dataset_root / "Train" / "Train.csv"
    split_path = dataset_root / "split.txt"
    names = datasets or discover_dataset_names(dataset_root)
    test_datasets = build_test_datasets(dataset_root, representation_root, names)

    train_params = TrainParams(test_datasets=test_datasets, output_root=output_root, model_name=model_name)
    train_params.FOLD_SEED = 42
    train_params.SPLIT_PATH = str(split_path)
    setup_runtime()
    Path(train_params.PREDICTION_CSV_DIR).mkdir(parents=True, exist_ok=True)

    training_cache = build_training_cache(
        train_params,
        residue_graph_root,
        cluster_root,
        residue_train_root=str(residue_train_root),
        token_train_root=str(token_train_root),
        label_file=str(label_file),
    )
    residue_sample = (
        load_sample_residue_graph(training_cache)
        if train_params.USE_EGNN_BRANCH
        else None
    )
    token_sample = load_sample_token(training_cache) if train_params.USE_IT_BRANCH else None
    residue_in_dim = residue_sample.x.size(1) if residue_sample is not None else None
    token_in_dim = token_sample.x.size(1) if token_sample is not None else None

    collate_fn = build_collate_fn()
    loader_kwargs = build_loader_kwargs(train_params.NUM_WORKERS, collate_fn)
    eval_loader_kwargs = build_loader_kwargs(train_params.NUM_WORKERS, collate_fn)

    _run_crossval_for_seed(
        train_params=train_params,
        training_cache=training_cache,
        residue_in_dim=residue_in_dim,
        token_in_dim=token_in_dim,
        residue_graph_root=residue_graph_root,
        cluster_root=cluster_root,
        loader_kwargs=loader_kwargs,
        eval_loader_kwargs=eval_loader_kwargs,
    )

if __name__ == "__main__":
    args = parse_args()
    train_model(
        dataset_root=args.dataset_root,
        representation_root=args.representation_root,
        output_root=args.output_root,
        model_name=args.model_name,
        datasets=args.datasets,
    )
