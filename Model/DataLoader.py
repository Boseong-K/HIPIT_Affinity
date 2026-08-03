import importlib
import sys
from functools import lru_cache
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader as TorchDataLoader, Dataset
from torch_geometric.data import Batch, Data
from Utils import read_label_map

def _extract_graph(graph_obj):
    if isinstance(graph_obj, dict):
        return graph_obj["graph"]
    return graph_obj

def resolve_graph_payload(graph_obj):
    optional_graph_dtypes = {"cluster": torch.long}
    keep_graph_attrs = {"x", "pos", "edge_index"} | set(optional_graph_dtypes)
    graph = _extract_graph(graph_obj)
    for key in list(graph.keys()):
        if key not in keep_graph_attrs:
            delattr(graph, key)
    graph.x = graph.x.float()
    graph.pos = graph.pos.float()
    graph.edge_index = graph.edge_index.long()
    for attr, dtype in optional_graph_dtypes.items():
        value = getattr(graph, attr, None)
        if torch.is_tensor(value):
            setattr(graph, attr, value.to(dtype=dtype))
    return graph

@lru_cache(maxsize=1)
def _patch_numpy_pickle_modules():
    for name in ("", "multiarray", "numeric", "umath", "fromnumeric", "shape_base"):
        source = "numpy.core" if not name else f"numpy.core.{name}"
        target = "numpy._core" if not name else f"numpy._core.{name}"
        try:
            sys.modules.setdefault(target, importlib.import_module(source))
        except Exception:
            pass

def _torch_load_cpu(path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")

@lru_cache(maxsize=None)
def _collect_input_paths(input_dir):
    paths_by_id = {}
    for path in sorted(Path(input_dir).rglob("*.pt")):
        paths_by_id.setdefault(path.stem.upper(), str(path))
    return paths_by_id

@lru_cache(maxsize=None)
def _read_label_map_cached(label_file, id_col, value_col):
    label_path = Path(label_file)
    return read_label_map(str(label_path), id_col, value_col)

def _cluster_split_name_for_graph(graph_path, residue_graph_root):
    rel = Path(graph_path).resolve().relative_to(Path(residue_graph_root).resolve())
    parts = rel.parts
    if parts[0] == "Fold":
        return "Train"
    return parts[0]

def _cluster_path_for_graph(graph_path, residue_graph_root, cluster_root):
    return Path(cluster_root) / _cluster_split_name_for_graph(graph_path, residue_graph_root) / Path(graph_path).name

def _attach_cached_clusters(graph, graph_path, residue_graph_root, cluster_root):
    cluster_path = _cluster_path_for_graph(graph_path, residue_graph_root, cluster_root)
    payload = _torch_load_cpu(cluster_path)
    cluster = payload.get("cluster")
    if torch.is_tensor(cluster):
        graph.cluster = cluster.long()
    return graph

def load_residue_graph(path, residue_graph_root, cluster_root):
    _patch_numpy_pickle_modules()
    payload = _torch_load_cpu(path)
    graph = resolve_graph_payload(payload)
    return _attach_cached_clusters(graph, path, residue_graph_root, cluster_root)

def load_token(path):
    _patch_numpy_pickle_modules()
    payload = _torch_load_cpu(path)
    tokens = payload["tokens"]
    x = tokens["x"].float()
    pos_value = tokens.get("pos")
    pos = pos_value.float()
    token_data = Data(x=x, pos=pos, edge_index=torch.empty((2, 0), dtype=torch.long))
    segment = tokens.get("segment")
    if torch.is_tensor(segment):
        token_data.segment = segment.long()
    return token_data

class TrainingDataCache:
    def __init__(
        self,
        params,
        residue_graph_dir,
        token_dir,
        label_map,
        residue_graph_root,
        cluster_root,
    ):
        self.residue_graph_root = residue_graph_root
        self.cluster_root = cluster_root
        self.use_egnn_branch = bool(params.USE_EGNN_BRANCH and residue_graph_dir)
        self.use_it_branch = bool(params.USE_IT_BRANCH and token_dir)
        self.residue_by_id = {}
        self.token_by_id = {}
        matched_ids = None
        if self.use_egnn_branch:
            self.residue_by_id = _collect_input_paths(str(Path(residue_graph_dir)))
            matched_ids = set(self.residue_by_id.keys())
        if self.use_it_branch:
            self.token_by_id = _collect_input_paths(str(Path(token_dir)))
            token_ids = set(self.token_by_id.keys())
            matched_ids = token_ids if matched_ids is None else matched_ids & token_ids
        self.label_map = label_map or {}
        self.ids = [
            pdb_id
            for pdb_id in sorted(matched_ids or [])
            if pdb_id in self.label_map and np.isfinite(self.label_map[pdb_id])
        ]
        self.labels = {pdb_id: self.label_map[pdb_id] for pdb_id in self.ids}

    def _get_residue_graph(self, pdb_id):
        path = self.residue_by_id[pdb_id]
        return load_residue_graph(path, self.residue_graph_root, self.cluster_root)

    def _get_token(self, pdb_id):
        return load_token(self.token_by_id[pdb_id])

    def item(self, pdb_id):
        sample = {
            "y": torch.tensor(self.labels[pdb_id], dtype=torch.float32),
            "pdb_id": pdb_id,
        }
        if self.use_egnn_branch:
            sample["residue_graph"] = self._get_residue_graph(pdb_id)
        if self.use_it_branch:
            sample["token"] = self._get_token(pdb_id)
        return sample

class ResidueTokenDataset(Dataset):
    def __init__(self, cache, ids):
        self.cache = cache
        self.ids = [str(pdb_id).upper() for pdb_id in ids]

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        return self.cache.item(self.ids[idx])

def build_collate_fn():
    def _collate(batch):
        residue_graphs = [item["residue_graph"] for item in batch if "residue_graph" in item]
        tokens = [item["token"] for item in batch if "token" in item]
        labels = torch.stack([item["y"] for item in batch], dim=0).float()
        return {
            "residue_graph": Batch.from_data_list(residue_graphs) if residue_graphs else None,
            "token": Batch.from_data_list(tokens) if tokens else None,
            "y": labels,
            "pdb_id": [item["pdb_id"] for item in batch],
        }

    return _collate

def build_loader(params, dataset, shuffle, loader_kwargs):
    generator = torch.Generator()
    generator.manual_seed(int(params.FOLD_SEED))
    return TorchDataLoader(
        dataset,
        batch_size=params.BATCH_SIZE,
        shuffle=shuffle,
        generator=generator,
        **loader_kwargs,
    )

def move_batch_to_device(batch, device):
    residue_graph_batch = batch["residue_graph"].to(device, non_blocking=True) if batch["residue_graph"] is not None else None
    token_batch = batch["token"].to(device, non_blocking=True) if batch["token"] is not None else None
    labels = batch["y"].to(device, non_blocking=True).view(-1)
    return residue_graph_batch, token_batch, labels

def build_residue_token_dataset(
    params,
    residue_graph_dir,
    token_dir,
    label_file,
    id_col,
    value_col,
    residue_graph_root,
    cluster_root,
):
    cache = TrainingDataCache(
        params=params,
        residue_graph_dir=residue_graph_dir,
        token_dir=token_dir,
        label_map=_read_label_map_cached(label_file, id_col, value_col),
        residue_graph_root=residue_graph_root,
        cluster_root=cluster_root,
    )
    return ResidueTokenDataset(cache, cache.ids)

def build_training_cache(
    params,
    residue_graph_root,
    cluster_root,
    residue_train_root,
    token_train_root,
    label_file,
    label_id_col="PDB",
    label_value_col="pKD",
):
    return TrainingDataCache(
        params=params,
        residue_graph_dir=residue_train_root,
        token_dir=token_train_root,
        label_map=_read_label_map_cached(label_file, label_id_col, label_value_col),
        residue_graph_root=residue_graph_root,
        cluster_root=cluster_root,
    )

def _seeded_fold_ids(params, ids, fold_idx, num_folds):
    rng = np.random.default_rng(int(params.FOLD_SEED))
    shuffled = list(rng.permutation(ids))
    folds = np.array_split(np.array(shuffled, dtype=object), num_folds)
    val_ids = set(str(pdb_id) for pdb_id in folds[fold_idx - 1].tolist())
    train_ids = [pdb_id for pdb_id in shuffled if pdb_id not in val_ids]
    return train_ids, sorted(val_ids)

@lru_cache(maxsize=None)
def _read_split_file(path):
    train_ids, val_ids = [], []
    for line in Path(path).read_text().splitlines():
        parts = line.strip().split()
        if len(parts) < 2:
            continue
        pdb_id, role = parts[0].upper(), parts[1].lower()
        if role.startswith("train"):
            train_ids.append(pdb_id)
        elif role.startswith("val"):
            val_ids.append(pdb_id)
    return train_ids, val_ids

def _list_based_ids(params, ids):
    available = set(str(pdb_id).upper() for pdb_id in ids)
    train_ids, val_ids = _read_split_file(params.SPLIT_PATH)
    return [i for i in train_ids if i in available], [i for i in val_ids if i in available]

def build_fold_dataset(training_cache, params, fold_idx, split_name, num_folds):
    if params.USE_CROSS_VALIDATION:
        train_ids, val_ids = _seeded_fold_ids(params, training_cache.ids, fold_idx, num_folds)
    else:
        train_ids, val_ids = _list_based_ids(params, training_cache.ids)
    selected_ids = train_ids if split_name == "Train" else val_ids
    return ResidueTokenDataset(training_cache, selected_ids)

def fold_input_dir(input_root, fold_idx):
    input_root = Path(input_root)
    fold_dir = input_root / f"Fold_{fold_idx}"
    return str(fold_dir if fold_dir.is_dir() else input_root)

def build_test_loaders(params, loader_kwargs, fold_idx, residue_graph_root, cluster_root):
    test_loaders = {}
    for dataset_name, cfg in params.TEST_DATASETS.items():
        residue_graph_dir = cfg.get("RESIDUE_DATA_ROOT")
        token_dir = fold_input_dir(cfg.get("TOKEN_DATA_ROOT"), fold_idx) if params.USE_IT_BRANCH else None
        test_set = build_residue_token_dataset(
            params=params,
            residue_graph_dir=residue_graph_dir,
            token_dir=token_dir,
            label_file=cfg.get("LABEL_FILE"),
            id_col=cfg.get("LABEL_ID_COL", "PDB"),
            value_col=cfg.get("LABEL_VALUE_COL", "pKD"),
            residue_graph_root=residue_graph_root,
            cluster_root=cluster_root,
        )
        test_loaders[dataset_name] = build_loader(params, test_set, False, loader_kwargs)
    return test_loaders

def load_sample_residue_graph(training_cache):
    return training_cache._get_residue_graph(training_cache.ids[0])

def load_sample_token(training_cache):
    return training_cache._get_token(training_cache.ids[0])
