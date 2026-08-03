from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from DataLoader import move_batch_to_device, build_loader
from Predictor import _build_model
from Utils import (
    RegressionMeter,
    autocast_context,
    snapshot_model_state,
    use_amp,)
from torch.utils.tensorboard import SummaryWriter

def _fold_prefix(fold_idx):
    return "" if fold_idx == "fixed" else f"Fold {fold_idx} "

def _is_no_decay_param(param):
    return param.ndim <= 1

def _build_optimizer_param_groups(model: nn.Module, lr: float, weight_decay: float):
    decay_params = []
    no_decay_params = []
    for param in model.parameters():
        if not param.requires_grad:
            continue
        if _is_no_decay_param(param):
            no_decay_params.append(param)
        else:
            decay_params.append(param)
    groups = []
    if decay_params:
        groups.append({"params": decay_params, "lr": float(lr), "weight_decay": float(weight_decay)})
    if no_decay_params:
        groups.append({"params": no_decay_params, "lr": float(lr), "weight_decay": 0.0})
    return groups

def build_optimizer(
    model: nn.Module,
    lr: float,
    weight_decay: float,
    device: torch.device,
    use_adamw=True,
):
    optimizer_cls = optim.AdamW if use_adamw else optim.Adam
    param_groups = _build_optimizer_param_groups(model, lr, weight_decay)
    if device.type == "cuda":
        try:
            return optimizer_cls(param_groups, fused=True)
        except TypeError:
            pass
    return optimizer_cls(param_groups)

def _train_one_epoch(train_loader, model, criterion, optimizer, scaler, device, fold_idx, epoch_idx):
    model.train()
    total_loss = 0.0
    tqdm_bar_format = "{desc}:{percentage:3.0f}%|{bar}|{n_fmt}/{total_fmt}[{elapsed}<{remaining},{rate_fmt}{postfix}]"
    loop = tqdm(
        train_loader,
        desc=f"{_fold_prefix(fold_idx)}Epoch {epoch_idx + 1}",
        leave=False,
        bar_format=tqdm_bar_format,
    )
    for batch in loop:
        residue_graph_batch, token_batch, labels = move_batch_to_device(batch, device)
        optimizer.zero_grad(set_to_none=True)
        with autocast_context(device):
            preds = model(residue_graph_batch, token_batch).view(-1)
        loss = criterion(preds.float(), labels.float())
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        loss_value = loss.item()
        total_loss += loss_value
        loop.set_postfix(loss=f"{loss_value:.4f}")
    return total_loss / max(1, len(train_loader))

def evaluate_model(loader, model, criterion, device, return_outputs=False):
    model.eval()
    total_loss = 0.0
    meter = RegressionMeter(store_outputs=return_outputs)

    with torch.inference_mode():
        for batch in loader:
            residue_graph_batch, token_batch, labels = move_batch_to_device(batch, device)
            with autocast_context(device):
                preds = model(residue_graph_batch, token_batch).view(-1)
            loss = criterion(preds.float(), labels.float())
            total_loss += loss.item()
            meter.update(preds, labels, ids=batch["pdb_id"] if return_outputs else None)
    return meter.compute(total_loss / max(1, len(loader)))

def _evaluate_named_loaders(test_loaders, model, criterion, device, return_outputs=False):
    return {
        dataset_name: evaluate_model(test_loader, model, criterion, device, return_outputs=return_outputs)
        for dataset_name, test_loader in test_loaders.items()
    }

def _evaluate_epoch(val_loader, model, criterion, device, epoch_idx, writer, train_mse):
    val_outputs = evaluate_model(val_loader, model, criterion, device, return_outputs=False)
    writer.add_scalar("Train/BatchMSE", train_mse, epoch_idx)
    _log_eval_outputs(writer, "Val", val_outputs, epoch_idx)
    return val_outputs

def _extract_test_metrics(test_outputs_by_name):
    return {
        dataset_name: {
            "rmse": outputs["rmse"],
            "mae": outputs["mae"],
            "r": outputs["r"],
            "r2": outputs["r2"],
        }
        for dataset_name, outputs in test_outputs_by_name.items()
    }

def _test_sizes_text(test_loaders):
    if not test_loaders:
        return "none"
    return ", ".join(f"{dataset_name}={len(test_loader.dataset)}" for dataset_name, test_loader in test_loaders.items())

def _metric_summary_text(name, outputs):
    return (
        f"{name} RMSE={outputs['rmse']:.4f} "
        f"MAE={outputs['mae']:.4f} "
        f"R={outputs['r']:.4f} "
        f"R2={outputs['r2']:.4f}"
    )

def _log_epoch_summary(fold_idx, epoch_idx, train_mse, val_outputs):
    tqdm.write(
        f"{_fold_prefix(fold_idx)}Epoch {epoch_idx + 1}: Train MSE={train_mse:.4f}|"
        + _metric_summary_text("Val", val_outputs)
    )

def _log_best_epoch_summary(fold_idx, best_epoch, best_val_outputs, test_outputs_by_name):
    prefix = _fold_prefix(fold_idx)
    tqdm.write(
        f"{prefix}Best epoch={best_epoch}|"
        f"BestVal RMSE={best_val_outputs['rmse']:.4f} "
        f"MAE={best_val_outputs['mae']:.4f} "
        f"R={best_val_outputs['r']:.4f} "
        f"R2={best_val_outputs['r2']:.4f}"
    )
    groups = _available_groups(test_outputs_by_name)
    if groups:
        for group_name, members in groups:
            metrics, group_size = _group_metrics_from_outputs(test_outputs_by_name, members)
            tqdm.write(f"{prefix}{_metric_summary_text(f'{group_name:<8}', metrics)}   (n={group_size})")
    else:
        for dataset_name, outputs in test_outputs_by_name.items():
            tqdm.write(f"{prefix}{_metric_summary_text(dataset_name, outputs)}")

def _mean_metric(fold_results, value_fn):
    values = [float(value_fn(fold_result)) for fold_result in fold_results]
    return float(np.mean(values)) if values else 0.0

def _available_groups(test_outputs_by_name):
    return [(name, (name,)) for name in test_outputs_by_name]

def _group_metrics_from_outputs(test_outputs_by_name, members):
    meter = RegressionMeter(store_outputs=False)
    for name in members:
        outputs = test_outputs_by_name[name]
        meter.update(torch.as_tensor(outputs["preds"]), torch.as_tensor(outputs["labels"]))
    return meter.compute(0.0), meter.count

def _group_metrics(fold_result, members):
    return _group_metrics_from_outputs(fold_result["test_outputs"], members)

def _log_eval_outputs(writer, prefix, outputs, epoch):
    writer.add_scalar(f"{prefix}/MSE", outputs["mse"], epoch)
    writer.add_scalar(f"{prefix}/RMSE", outputs["rmse"], epoch)
    writer.add_scalar(f"{prefix}/MAE", outputs["mae"], epoch)
    writer.add_scalar(f"{prefix}/R2", outputs["r2"], epoch)
    writer.add_scalar(f"{prefix}/PearsonR", outputs["r"], epoch)

def _branch_usage_text(train_params):
    if train_params.USE_EGNN_BRANCH and train_params.USE_IT_BRANCH:
        return "egnn=True it=True fusion=cross_attention"
    if train_params.USE_EGNN_BRANCH:
        return "egnn=True it=False mode=single"
    if train_params.USE_IT_BRANCH:
        return "egnn=False it=True mode=single"
    return f"egnn={train_params.USE_EGNN_BRANCH} it={train_params.USE_IT_BRANCH}"

def _log_fold_start(train_params, fold_idx, train_dataset, val_dataset, test_loaders):
    prefix = _fold_prefix(fold_idx)
    tqdm.write(f"Branch usage: {_branch_usage_text(train_params)}")
    tqdm.write(f"{prefix}external test sizes: {_test_sizes_text(test_loaders)}")
    tqdm.write(
        f"{prefix}train={len(train_dataset)} "
        f"val={len(val_dataset)} checkpoint={train_params.MODEL_STATE_PATH}"
    )

def _arch_config(model, train_params, learning_rate, weight_decay):
    cfg = {
        "use_egnn_branch": bool(train_params.USE_EGNN_BRANCH),
        "use_it_branch": bool(train_params.USE_IT_BRANCH),
        "batch_size": int(train_params.BATCH_SIZE),
        "learning_rate": float(learning_rate),
        "weight_decay": float(weight_decay),
        "fold_seed": int(train_params.FOLD_SEED),
    }
    egnn = getattr(model, "egnn_encoder", None)
    if egnn is not None:
        cfg["egnn"] = {
            "num_layers": int(egnn.num_layers),
            "use_hip": bool(egnn.use_hip),
            "pool_after_layers": int(egnn.pool_after_layers),
            "hidden_dim": int(egnn.hidden_dim),
            "out_dim": int(egnn.out_dim),
        }
    it = getattr(model, "it_encoder", None)
    if it is not None:
        cfg["it"] = {
            "num_layers": int(getattr(it, "num_layers", -1)),
            "out_dim": int(getattr(it, "out_dim", -1)),
        }
    return cfg

def _run_single_fold(
    train_params,
    fold_idx,
    train_dataset,
    val_dataset,
    residue_in_dim,
    token_in_dim,
    loader_kwargs,
    eval_loader_kwargs,
    test_loaders,
):
    fold_log_dir = Path(train_params.LOG_DIR)
    fold_log_dir.mkdir(parents=True, exist_ok=True)
    Path(train_params.MODEL_STATE_PATH).parent.mkdir(parents=True, exist_ok=True)

    learning_rate = 1e-5
    weight_decay = 1e-4
    train_loader = build_loader(train_params, train_dataset, True, loader_kwargs)
    val_loader = build_loader(train_params, val_dataset, False, eval_loader_kwargs)
    criterion = nn.MSELoss()
    model = _build_model(train_params, residue_in_dim, token_in_dim)
    arch_cfg = _arch_config(model, train_params, learning_rate, weight_decay)
    optimizer = build_optimizer(
        model,
        lr=learning_rate,
        weight_decay=weight_decay,
        device=train_params.DEVICE,
        use_adamw=True,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp(train_params.DEVICE))
    writer = SummaryWriter(log_dir=fold_log_dir)

    best_model_state = snapshot_model_state(model)
    best_val_rmse = float("inf")
    best_epoch = 0
    epochs_without_improvement = 0
    start_epoch = 0

    checkpoint_path = Path(train_params.MODEL_STATE_PATH)
    if checkpoint_path.exists():
        ckpt = torch.load(checkpoint_path, map_location=train_params.DEVICE, weights_only=False)
        if ckpt.get("done"):
            writer.close()
            tqdm.write(f"{_fold_prefix(fold_idx)}already complete — skipping (loaded saved result)")
            return ckpt["fold_result"]
        model.load_state_dict(ckpt["model_state"])
        optimizer.load_state_dict(ckpt["optimizer_state"])
        scaler.load_state_dict(ckpt["scaler_state"])
        best_model_state = ckpt["best_model_state"]
        best_val_rmse = ckpt["best_val_rmse"]
        best_epoch = ckpt["best_epoch"]
        epochs_without_improvement = ckpt["epochs_without_improvement"]
        start_epoch = ckpt["epoch"] + 1
        tqdm.write(f"{_fold_prefix(fold_idx)}resumed at epoch {start_epoch} (best epoch={best_epoch}, best val RMSE={best_val_rmse:.4f})")

    for epoch in range(start_epoch, train_params.EPOCHS):
        train_mse = _train_one_epoch(
            train_loader,
            model,
            criterion,
            optimizer,
            scaler,
            train_params.DEVICE,
            fold_idx,
            epoch,
        )
        val_outputs = _evaluate_epoch(
            val_loader,
            model,
            criterion,
            train_params.DEVICE,
            epoch,
            writer,
            train_mse,
        )
        writer.add_scalar("Train/LR", optimizer.param_groups[0]["lr"], epoch)
        val_rmse = val_outputs["rmse"]
        _log_epoch_summary(fold_idx, epoch, train_mse, val_outputs)

        if val_rmse < best_val_rmse:
            best_val_rmse = val_rmse
            best_epoch = epoch + 1
            best_model_state = snapshot_model_state(model)
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        torch.save(
            {
                "best_model_state": best_model_state,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "scaler_state": scaler.state_dict(),
                "epoch": epoch,
                "best_val_rmse": best_val_rmse,
                "best_epoch": best_epoch,
                "epochs_without_improvement": epochs_without_improvement,
                "config": arch_cfg,
                "done": False,
            },
            train_params.MODEL_STATE_PATH,
        )

        if epochs_without_improvement >= 5:
            break
    model.load_state_dict(best_model_state)
    best_val_outputs = evaluate_model(
        val_loader,
        model,
        criterion,
        train_params.DEVICE,
        return_outputs=True,
    )

    test_outputs_by_name = _evaluate_named_loaders(test_loaders, model, criterion, train_params.DEVICE, return_outputs=True)
    test_metrics_by_name = _extract_test_metrics(test_outputs_by_name)
    if not train_params.USE_CROSS_VALIDATION:
        _log_best_epoch_summary(fold_idx, best_epoch, best_val_outputs, test_outputs_by_name)
    writer.close()
    fold_result = {
        "fold": fold_idx,
        "train_size": len(train_dataset),
        "val_size": len(val_dataset),
        "best_epoch": best_epoch,
        "best_val_rmse": best_val_outputs["rmse"],
        "best_val_mae": best_val_outputs["mae"],
        "best_val_r": best_val_outputs["r"],
        "best_val_r2": best_val_outputs["r2"],
        "test_metrics": test_metrics_by_name,
        "test_outputs": test_outputs_by_name,
    }
    torch.save(
        {"best_model_state": best_model_state, "config": arch_cfg, "done": True, "fold_result": fold_result},
        train_params.MODEL_STATE_PATH,
    )
    return fold_result
