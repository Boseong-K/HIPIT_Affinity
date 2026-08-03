import torch
import torch.nn as nn
from torch_geometric.utils import coalesce, remove_self_loops, scatter, softmax

class HierarchicalInterfaceAwarePooling(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.in_dim = in_dim
        self.attn_proj = nn.Linear(in_dim, 1)
        self.pool_norm = nn.LayerNorm(in_dim)

    def forward(self, h, pos, edge_index, batch, cluster):
        num_clusters = int(cluster.max().item()) + 1
        attn_weights = softmax(self.attn_proj(h), cluster, num_nodes=num_clusters)
        pooled_h = scatter(attn_weights * h, cluster, dim=0, dim_size=num_clusters, reduce="sum")
        pooled_h = self.pool_norm(pooled_h).to(h.dtype)
        pooled_edge_index = coalesce(cluster[edge_index], num_nodes=num_clusters)
        pooled_edge_index, _ = remove_self_loops(pooled_edge_index)
        pooled_batch = h.new_empty(num_clusters, dtype=torch.long)
        pooled_batch = pooled_batch.scatter_(0, cluster, batch)
        pooled_pos = scatter(pos, cluster, dim=0, dim_size=num_clusters, reduce="mean")
        return pooled_h, pooled_pos, pooled_edge_index, pooled_batch

    def __repr__(self):
        return f"{self.__class__.__name__}({self.in_dim})"
