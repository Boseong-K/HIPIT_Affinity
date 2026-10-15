import argparse
import csv
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import HIPIT_Affinity as H
from DataLoader import (
    _collect_input_paths,
    build_collate_fn,
    build_loader,
    build_residue_token_dataset,
    load_residue_graph,
    load_token,)
from Predictor import _build_model
from Training import evaluate_model
from Utils import build_loader_kwargs, setup_runtime

def parse_args():
    p = argparse.ArgumentParser(description="HIP-IT inference on one dataset.")
    p.add_argument("--checkpoint", required=True, help="trained .pt checkpoint")
    p.add_argument("--input", required=True, help="dataset name (subdir under --representation-root)")
    p.add_argument("--labels", required=True, help="label CSV with columns PDB, pKD")
    p.add_argument("--representation-root", default=str(H.REPRESENTATION_ROOT),
                   help="root holding Residue_Graph/ Atom_Token/ Interface_Cluster/")
    p.add_argument("--id-col", default="PDB")
    p.add_argument("--value-col", default="pKD")
    p.add_argument("--device", default="cuda:1")
    p.add_argument("--output", default=None, help="predictions CSV path (default: <ckpt>_<input>_predictions.csv)")
    return p.parse_args()

def main():
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    setup_runtime()

    ck = torch.load(args.checkpoint, map_location=device, weights_only=False)
    conf = ck.get("config", {}) if isinstance(ck, dict) else {}
    use_egnn = bool(conf.get("use_egnn_branch", True))
    use_it = bool(conf.get("use_it_branch", True))

    rep = Path(args.representation_root)
    residue_graph_root = rep / "Residue_Graph"
    cluster_root = rep / "Interface_Cluster"
    residue_dir = residue_graph_root / args.input
    token_dir = rep / "Atom_Token" / args.input

    params = H.TrainParams(test_datasets={})
    params.DEVICE = device
    params.FOLD_SEED = 42
    params.USE_EGNN_BRANCH = use_egnn
    params.USE_IT_BRANCH = use_it

    res_dim = tok_dim = None
    if use_egnn:
        paths = _collect_input_paths(str(residue_dir))
        res_dim = load_residue_graph(paths[sorted(paths)[0]], residue_graph_root, cluster_root).x.size(1)
    if use_it:
        paths = _collect_input_paths(str(token_dir))
        tok_dim = load_token(paths[sorted(paths)[0]]).x.size(1)

    model = _build_model(params, res_dim, tok_dim)
    state = ck["best_model_state"] if isinstance(ck, dict) and "best_model_state" in ck else ck
    model.load_state_dict(state)
    model.eval()
    print(f"[infer] {Path(args.checkpoint).name} | egnn={use_egnn} it={use_it} | input={args.input} | {device}", flush=True)

    dataset = build_residue_token_dataset(
        params=params,
        residue_graph_dir=str(residue_dir),
        token_dir=str(token_dir) if use_it else None,
        label_file=args.labels,
        id_col=args.id_col,
        value_col=args.value_col,
        residue_graph_root=residue_graph_root,
        cluster_root=cluster_root,
    )
    loader = build_loader(params, dataset, False, build_loader_kwargs(params.NUM_WORKERS, build_collate_fn()))
    out = evaluate_model(loader, model, nn.MSELoss(), device, return_outputs=True)

    ids = out["ids"]
    labels = np.asarray(out["labels"], dtype=float).ravel()
    preds = np.asarray(out["preds"], dtype=float).ravel()
    print(f"\nN = {len(ids)}")
    for k in ("rmse", "mae", "r", "r2"):
        print(f"  {k.upper():4} = {out[k]:.4f}")

    out_path = args.output or f"{Path(args.checkpoint).stem}_{args.input}_predictions.csv"
    with open(out_path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["PDB", "Actual", "Predicted"])
        for pdb, a, p in zip(ids, labels, preds):
            w.writerow([pdb, f"{a:.6f}", f"{p:.6f}"])
    print(f"\nwrote {out_path}", flush=True)

if __name__ == "__main__":
    main()
