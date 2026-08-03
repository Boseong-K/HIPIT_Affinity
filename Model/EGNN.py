import torch
from torch import nn
from torch_geometric.utils import scatter
from Hierarchical_Interface_aware_Pooling import HierarchicalInterfaceAwarePooling

class RBFEncoder(nn.Module):
    def __init__(self, num_rbf, cutoff):
        super().__init__()
        num_rbf = int(num_rbf)
        cutoff = float(cutoff)
        centers = torch.linspace(0.0, cutoff, num_rbf)
        delta = centers[1] - centers[0]
        gamma = 1.0 / (delta * delta + 1e-12)
        self.centers = nn.Parameter(centers)
        self.log_gamma = nn.Parameter(torch.log(gamma.expand(num_rbf).clone()))

    def forward(self, dist):
        gamma = torch.exp(self.log_gamma).view(1, -1)
        centers = self.centers.view(1, -1)
        return torch.exp(-gamma * (dist - centers) ** 2)

class EGNNLayer(nn.Module):
    def __init__(
        self,
        dim,
        cutoff,
        dropout,
        num_alphacarbon_rbf,
    ):
        super().__init__()
        self.alphacarbon_rbf = RBFEncoder(num_rbf=num_alphacarbon_rbf, cutoff=cutoff)
        edge_rbf_dim = int(num_alphacarbon_rbf)
        self.edge_mlp = nn.Sequential(
            nn.Linear(2 * dim + edge_rbf_dim, dim),
            nn.SiLU(),
            nn.Linear(dim, dim),
            nn.SiLU(),
        )
        self.alphacarbon_coord_mlp = nn.Sequential(
            nn.Linear(dim, 1),
            nn.Tanh(),
        )
        nn.init.xavier_uniform_(self.alphacarbon_coord_mlp[0].weight, gain=0.001)
        nn.init.zeros_(self.alphacarbon_coord_mlp[0].bias)
        self.node_mlp = nn.Sequential(
            nn.Linear(dim + dim, dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
            nn.SiLU(),
        )

        self.node_norm = nn.LayerNorm(dim)
        self.alphacarbon_coord_res_weight = nn.Parameter(torch.tensor(0.1))

    def forward(self, h, pos, edge_index):
        row, col = edge_index
        num_edges = edge_index.size(1)
        alphacarbon_rel_pos = pos[row] - pos[col]
        alphacarbon_dist = torch.sqrt(alphacarbon_rel_pos.square().sum(dim=-1, keepdim=True) + 1e-8)
        alphacarbon_dist_feat = self.alphacarbon_rbf(alphacarbon_dist)
        alphacarbon_dist_feat = alphacarbon_dist_feat.to(device=h.device, dtype=h.dtype)
        edge_feat = torch.cat([h[row], h[col], alphacarbon_dist_feat], dim=-1)
        edge_message = self.edge_mlp(edge_feat)
        degree = torch.zeros(h.size(0), 1, device=h.device, dtype=h.dtype)
        degree.index_add_(0, row, h.new_ones(num_edges, 1))
        degree = degree.clamp_min_(1e-8)
        alphacarbon_scale = self.alphacarbon_coord_mlp(edge_message)
        alphacarbon_direction = alphacarbon_rel_pos / (alphacarbon_dist + 1e-8)
        alphacarbon_delta = torch.zeros_like(pos)
        alphacarbon_delta.index_add_(0, row, (alphacarbon_direction * alphacarbon_scale).to(alphacarbon_delta.dtype))
        pos = pos + self.alphacarbon_coord_res_weight * (alphacarbon_delta / degree.to(alphacarbon_delta.dtype))
        neighbor_message = scatter(edge_message.to(h.dtype), row, dim=0, dim_size=h.size(0), reduce="mean")
        h = h + self.node_mlp(torch.cat([h, neighbor_message], dim=-1))
        h = self.node_norm(h)
        return h, pos

class EGNN(nn.Module):
    def __init__(self, in_dim, use_hip=False):
        super().__init__()
        self.hidden_dim = 1024
        self.out_dim = 1024
        self.num_layers = 5
        self.use_hip = bool(use_hip)
        self.pool_after_layers = 3
        self.input_proj = nn.Linear(in_dim, self.hidden_dim)
        self.layers = nn.ModuleList(
            [
                EGNNLayer(
                    self.hidden_dim,
                    cutoff=30.0,
                    dropout=0.5,
                    num_alphacarbon_rbf=128,
                )
                for _ in range(self.num_layers)
            ]
        )
        self.output_proj = nn.Linear(self.hidden_dim, self.out_dim)
        self.readout_norm = nn.LayerNorm(self.out_dim)
        self.readout_dropout = nn.Dropout(0.0)
        self.hip_pool = HierarchicalInterfaceAwarePooling(self.hidden_dim) if self.use_hip else None

    def _max_pool(self, h, batch, num_graphs=None):
        if num_graphs is None:
            num_graphs = int(batch.max().item()) + 1 if batch.numel() > 0 else 0
        return scatter(h, batch, dim=0, dim_size=int(num_graphs), reduce="max")

    def _normalize_cluster(self, cluster, batch):
        cluster = cluster.view(-1).to(device=batch.device, dtype=torch.long)
        if cluster.numel() == 0:
            return cluster
        stride = cluster.max() + 1
        key = batch.to(dtype=torch.long) * stride + cluster
        _, normalized = torch.unique(key, sorted=True, return_inverse=True)
        return normalized

    def _pool_graph(self, h, pos, edge_index, batch, cluster=None):
        if cluster is None:
            raise SystemExit("HIP requires cluster input.")
        cluster = self._normalize_cluster(cluster, batch)
        if cluster.numel() == 0:
            return h, pos, edge_index, batch
        num_clusters = int(cluster.max().item()) + 1
        if num_clusters >= h.size(0):
            return h, pos, edge_index, batch
        h, pos, edge_index, batch = self.hip_pool(
            h,
            pos,
            edge_index,
            batch,
            cluster,
        )

        return h, pos, edge_index, batch

    def forward(self, x, pos, edge_index, batch, cluster=None, num_graphs=None):
        h = self.input_proj(x)
        if cluster is not None:
            cluster = cluster.to(device=batch.device, dtype=torch.long)
        has_pooled = False

        for layer_idx, layer in enumerate(self.layers):
            if self.use_hip and not has_pooled and layer_idx == self.pool_after_layers:
                h, pos, edge_index, batch = self._pool_graph(
                    h,
                    pos,
                    edge_index,
                    batch,
                    cluster=cluster,
                )
                has_pooled = True

            h, pos = layer(
                h,
                pos,
                edge_index,
            )

        if self.use_hip and not has_pooled:
            h, pos, edge_index, batch = self._pool_graph(
                h,
                pos,
                edge_index,
                batch,
                cluster=cluster,
            )

        rep_egnn = self._max_pool(h, batch, num_graphs=num_graphs)
        rep_egnn = self.output_proj(rep_egnn)
        rep_egnn = self.readout_norm(rep_egnn)
        rep_egnn = self.readout_dropout(rep_egnn)

        return rep_egnn

def build_egnn(in_dim, use_hip=True):
    return EGNN(in_dim=in_dim, use_hip=use_hip)
