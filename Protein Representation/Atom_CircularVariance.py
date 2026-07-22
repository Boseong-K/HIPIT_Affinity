from typing import Dict, Sequence
import numpy as np
from scipy.spatial import cKDTree

def get_circular_variance_dim() -> int:
    return 1

def _stack_coords(entries: Sequence[Dict[str, object]]) -> np.ndarray:
    if not entries:
        return np.zeros((0, 3), dtype=np.float64)
    return np.stack([np.asarray(entry["coord"], dtype=np.float64) for entry in entries], axis=0)

def circular_variance_features(
    context_entries: Sequence[Dict[str, object]],
    target_entries: Sequence[Dict[str, object]],
) -> np.ndarray:
    out = np.zeros((len(target_entries), 1), dtype=np.float32)
    if len(target_entries) == 0:
        return out
    ctx = _stack_coords(context_entries)
    tgt = _stack_coords(target_entries)
    if ctx.shape[0] == 0:
        return out
    tree = cKDTree(ctx)
    neighbor_lists = tree.query_ball_point(tgt, r=6.0)
    for i, nbrs in enumerate(neighbor_lists):
        if not nbrs:
            continue
        d = ctx[nbrs] - tgt[i]
        dist = np.sqrt((d * d).sum(axis=1))
        keep = dist > 1e-6
        if not keep.any():
            continue
        units = d[keep] / dist[keep][:, None]
        out[i, 0] = np.float32(1.0 - np.linalg.norm(units.mean(axis=0)))
    return out
