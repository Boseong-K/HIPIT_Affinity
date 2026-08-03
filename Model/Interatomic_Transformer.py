import math
import torch
from torch import nn
from torch_geometric.utils import to_dense_batch

def encode_radial_distance(dist, centers, gamma):
    centers = centers.to(device=dist.device, dtype=dist.dtype)
    gamma = gamma.to(device=dist.device, dtype=dist.dtype)
    return torch.exp(-gamma * (dist.unsqueeze(-1) - centers) ** 2)

def make_radial_distance_basis(num_rbf, cutoff):
    centers = torch.linspace(0.0, float(cutoff), int(num_rbf))
    delta = centers[1] - centers[0]
    gamma = 1.0 / (delta * delta + 1e-12)
    return centers, gamma

def init_distance_bias(projection, num_rbf, init_scale=1.0):
    with torch.no_grad():
        ramp = torch.linspace(1.0, -1.0, int(num_rbf))
        projection.weight.copy_(float(init_scale) * ramp.unsqueeze(0).expand_as(projection.weight))

def gather_neighbors(source, neighbor_index):
    batch_size, num_tokens, feat_dim = source.shape
    k = neighbor_index.size(-1)
    flat_index = neighbor_index.reshape(batch_size, num_tokens * k, 1).expand(batch_size, num_tokens * k, feat_dim)
    return source.gather(1, flat_index).reshape(batch_size, num_tokens, k, feat_dim)

class InterAtomicTransformerBlock(nn.Module):
    def __init__(self, dim, heads, ffn_ratio, attention_dropout, ffn_dropout,
                 attention_temperature, num_rbf, pair_dim=32):
        super().__init__()
        self.dim = int(dim)
        self.heads = int(heads)
        self.head_dim = self.dim // self.heads
        self.pair_dim = int(pair_dim)
        self.attention_temperature = float(attention_temperature)
        self.query_proj = nn.Linear(dim, self.pair_dim)
        self.key_proj = nn.Linear(dim, self.pair_dim)
        self.pair_proj = nn.Linear(self.pair_dim, self.heads)
        nn.init.constant_(self.pair_proj.bias, -2.0)
        self.value_proj = nn.Linear(dim, dim)
        self.distance_bias = nn.Linear(int(num_rbf), self.heads, bias=False)
        init_distance_bias(self.distance_bias, int(num_rbf), init_scale=1.0)
        self.out_proj = nn.Linear(dim, dim)
        self.attn_drop = nn.Dropout(float(attention_dropout))
        self.attn_norm = nn.LayerNorm(dim)
        self.ffn_norm = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * int(ffn_ratio)),
            nn.GELU(),
            nn.Dropout(float(ffn_dropout)),
            nn.Linear(dim * int(ffn_ratio), dim),
        )

    def _pair_logits(self, normed, neighbor_rbf, neighbor_index):
        with torch.autocast(device_type=normed.device.type, enabled=False):
            x = normed.float()
            query = self.query_proj(x).unsqueeze(2)
            key = gather_neighbors(self.key_proj(x), neighbor_index)
            interaction = torch.nn.functional.gelu(query * key)
            logit = self.pair_proj(interaction)
            distance_bias_term = self.distance_bias(neighbor_rbf.float())
            return logit + distance_bias_term

    def _attend(self, normed, neighbor_rbf, neighbor_index, neighbor_valid):
        batch_size, num_tokens, _ = normed.shape
        k = neighbor_index.size(-1)
        heads, head_dim = self.heads, self.head_dim
        value_nbr = gather_neighbors(self.value_proj(normed), neighbor_index).view(batch_size, num_tokens, k, heads, head_dim)
        logits = self._pair_logits(normed, neighbor_rbf, neighbor_index)
        block_mask = ~neighbor_valid.unsqueeze(-1)
        with torch.autocast(device_type=normed.device.type, enabled=False):
            gate = torch.sigmoid(logits / self.attention_temperature)
            gate = self.attn_drop(gate.masked_fill(block_mask, 0.0))
            out = torch.einsum("bnkh,bnkhd->bnhd", gate, value_nbr.float()) / math.sqrt(k)
        return out.reshape(batch_size, num_tokens, self.dim)

    def forward(self, tokens, mask, neighbor_rbf=None, neighbor_index=None, neighbor_valid=None):
        attn_output = self._attend(tokens, neighbor_rbf, neighbor_index, neighbor_valid)
        tokens = self.attn_norm(tokens + self.out_proj(attn_output))
        tokens = self.ffn_norm(tokens + self.ffn(tokens))
        return tokens.masked_fill(~mask.unsqueeze(-1), 0)

class InterAtomicTransformer(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        hidden_dim = 1024
        in_dim = int(in_dim)
        self.hidden_dim = hidden_dim
        self.local_k = 32
        self.input_proj = nn.Linear(in_dim, hidden_dim)
        self.num_layers = 4
        self.layers = nn.ModuleList(
            InterAtomicTransformerBlock(
                hidden_dim, heads=1, ffn_ratio=4, attention_dropout=0.0, ffn_dropout=0.0,
                attention_temperature=0.5, num_rbf=64, pair_dim=32,
            )
            for _ in range(self.num_layers)
        )
        self.out_dim = hidden_dim
        centers, gamma = make_radial_distance_basis(64, 8.0)
        self.register_buffer("distance_centers", centers)
        self.register_buffer("distance_gamma", gamma)

    def _prepare_token_inputs(self, x, pos, batch):
        x, mask = to_dense_batch(x, batch=batch, fill_value=0)
        pos, _ = to_dense_batch(pos, batch=batch, fill_value=0)
        return x, pos.float(), mask

    def _distance_rbf(self, neighbor_dist, dtype):
        dist = neighbor_dist.to(dtype=dtype).clamp_min(0.0)
        return encode_radial_distance(dist, self.distance_centers, self.distance_gamma)

    def _cross_local_neighbors(self, pair_dist, mask, segment):
        n = pair_dist.size(1)
        cand = (segment.unsqueeze(2) != segment.unsqueeze(1)) & mask.unsqueeze(1)
        d = pair_dist.masked_fill_(~cand, float("inf"))
        k = min(self.local_k, n)
        neighbor_dist, neighbor_index = d.topk(k, dim=-1, largest=False)
        neighbor_valid = torch.isfinite(neighbor_dist)
        return neighbor_index, neighbor_valid, neighbor_dist

    def forward(self, x, pos, batch, segment=None):
        x, pos, mask = self._prepare_token_inputs(x, pos, batch)
        token_features = x.to(dtype=torch.float32)
        pair_dist = torch.cdist(pos, pos)
        segment_dense, _ = to_dense_batch(segment, batch=batch, fill_value=-1)
        neighbor_index, neighbor_valid, neighbor_dist = self._cross_local_neighbors(pair_dist, mask, segment_dense)
        tokens = self.input_proj(token_features).masked_fill(~mask.unsqueeze(-1), 0)
        neighbor_rbf = self._distance_rbf(neighbor_dist, tokens.dtype)
        for layer in self.layers:
            tokens = layer(tokens, mask, neighbor_rbf=neighbor_rbf,
                           neighbor_index=neighbor_index, neighbor_valid=neighbor_valid)
        rep_it = tokens.masked_fill(~mask.unsqueeze(-1), float("-inf")).max(dim=1).values
        return rep_it

def build_it_encoder(in_dim):
    return InterAtomicTransformer(in_dim=in_dim)
