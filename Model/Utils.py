from typing import Dict
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

class RegressionMeter:
    def __init__(self, store_outputs=False):
        self.store_outputs = bool(store_outputs)
        self.count = 0
        self.sum_abs = 0.0
        self.sum_sq = 0.0
        self.sum_pred = 0.0
        self.sum_label = 0.0
        self.sum_pred_sq = 0.0
        self.sum_label_sq = 0.0
        self.sum_cross = 0.0
        self.ids = [] if self.store_outputs else None
        self.preds = [] if self.store_outputs else None
        self.labels = [] if self.store_outputs else None

    def update(self, preds, labels, ids=None):
        preds_cpu = preds.detach().float().view(-1).cpu()
        labels_cpu = labels.detach().float().view(-1).cpu()
        error = preds_cpu - labels_cpu
        self.count += int(preds_cpu.numel())
        self.sum_abs += float(error.abs().sum().item())
        self.sum_sq += float((error * error).sum().item())
        self.sum_pred += float(preds_cpu.sum().item())
        self.sum_label += float(labels_cpu.sum().item())
        self.sum_pred_sq += float((preds_cpu * preds_cpu).sum().item())
        self.sum_label_sq += float((labels_cpu * labels_cpu).sum().item())
        self.sum_cross += float((preds_cpu * labels_cpu).sum().item())
        if self.store_outputs:
            self.preds.append(preds_cpu.numpy())
            self.labels.append(labels_cpu.numpy())
            if ids is not None:
                self.ids.extend(str(item).upper() for item in ids)

    def compute(self, mean_loss):
        if self.count == 0:
            outputs = {
                "mse": float(mean_loss),
                "rmse": 0.0,
                "mae": 0.0,
                "r2": 0.0,
                "r": 0.0,
            }
            if self.store_outputs:
                outputs["ids"] = []
                outputs["preds"] = np.zeros((0,), dtype=np.float32)
                outputs["labels"] = np.zeros((0,), dtype=np.float32)
            return outputs
        mean_label = self.sum_label / self.count
        ss_tot = self.sum_label_sq - self.count * mean_label * mean_label
        numerator = self.count * self.sum_cross - self.sum_pred * self.sum_label
        denom_pred = self.count * self.sum_pred_sq - self.sum_pred * self.sum_pred
        denom_label = self.count * self.sum_label_sq - self.sum_label * self.sum_label
        if denom_pred <= 1e-12 or denom_label <= 1e-12:
            r = 0.0
        else:
            r = float(numerator / np.sqrt(denom_pred * denom_label))
        outputs = {
            "mse": float(mean_loss),
            "rmse": float(np.sqrt(self.sum_sq / self.count)),
            "mae": float(self.sum_abs / self.count),
            "r2": 0.0 if ss_tot <= 1e-12 else float(1.0 - self.sum_sq / ss_tot),
            "r": r,
        }
        if self.store_outputs:
            outputs["ids"] = list(self.ids)
            outputs["preds"] = np.concatenate(self.preds, axis=0) if self.preds else np.zeros((0,), dtype=np.float32)
            outputs["labels"] = np.concatenate(self.labels, axis=0) if self.labels else np.zeros((0,), dtype=np.float32)
        return outputs

def _regression_metrics_np(labels, preds):
    err = preds - labels
    ss_res = float(np.sum(err * err))
    mean_label = float(labels.mean())
    ss_tot = float(np.sum((labels - mean_label) ** 2))
    r = float(np.corrcoef(labels, preds)[0, 1]) if labels.std() > 1e-12 and preds.std() > 1e-12 else 0.0
    return {
        "rmse": float(np.sqrt(np.mean(err * err))),
        "mae": float(np.mean(np.abs(err))),
        "r": r,
        "r2": float(1.0 - ss_res / ss_tot) if ss_tot > 1e-12 else 0.0,
    }

def bootstrap_metric_cis(labels, preds, n_boot=1000, seed=0, alpha=0.05):
    labels = np.asarray(labels, dtype=np.float64).ravel()
    preds = np.asarray(preds, dtype=np.float64).ravel()
    n = int(labels.shape[0])
    point = _regression_metrics_np(labels, preds)
    if n < 2:
        return {k: {"point": v, "lo": v, "hi": v} for k, v in point.items()}
    rng = np.random.default_rng(int(seed))
    samples = {k: [] for k in point}
    for _ in range(int(n_boot)):
        idx = rng.integers(0, n, n)
        m = _regression_metrics_np(labels[idx], preds[idx])
        for k in samples:
            samples[k].append(m[k])
    lo_q, hi_q = 100.0 * (alpha / 2.0), 100.0 * (1.0 - alpha / 2.0)
    return {
        k: {
            "point": point[k],
            "lo": float(np.nanpercentile(samples[k], lo_q)),
            "hi": float(np.nanpercentile(samples[k], hi_q)),
        }
        for k in point
    }

def read_label_map(label_file: str, id_col: str, value_col: str) -> Dict[str, float]:
    df = pd.read_csv(label_file)
    label_df = df[[id_col, value_col]].dropna()
    return {
        str(pdb_id).strip().upper(): float(value)
        for pdb_id, value in zip(label_df[id_col].to_numpy(), label_df[value_col].to_numpy())
    }

def snapshot_model_state(model: nn.Module) -> Dict[str, torch.Tensor]:
    return {
        key: value.detach().to("cpu", copy=True) if torch.is_tensor(value) else value
        for key, value in model.state_dict().items()
    }

def load_checkpoint_weights(path, map_location="cpu"):
    obj = torch.load(path, map_location=map_location, weights_only=False)
    if isinstance(obj, dict) and "best_model_state" in obj:
        return obj["best_model_state"]
    return obj

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def build_loader_kwargs(num_workers: int, collate_fn):
    num_workers = int(num_workers)
    loader_kwargs = dict(
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
        collate_fn=collate_fn,
    )
    if num_workers > 0:
        loader_kwargs["worker_init_fn"] = seed_worker
    return loader_kwargs

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def setup_runtime():
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

def use_amp(device):
    return device.type == "cuda"

def autocast_context(device: torch.device):
    device_type = "cuda" if device.type == "cuda" else "cpu"
    amp_dtype = torch.float16 if device_type == "cuda" else torch.bfloat16
    return torch.autocast(device_type=device_type, dtype=amp_dtype, enabled=use_amp(device))
