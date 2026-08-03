import torch
import torch.nn as nn

class CrossAttentionFusion(nn.Module):
    def __init__(self, egnn_dim):
        super().__init__()
        egnn_dim = int(egnn_dim)
        self.num_chunks = 8
        self.fusion_dim = egnn_dim
        self.fusion_norm = nn.LayerNorm(self.fusion_dim)
        self.chunk_width = egnn_dim // self.num_chunks
        self.egnn_cross_attn = nn.MultiheadAttention(
            embed_dim=self.chunk_width,
            num_heads=1,
            dropout=0.1,
            batch_first=True,
        )
        self.fusion_gate = nn.Parameter(torch.tensor(0.1))

    def _to_chunks(self, rep):
        return rep.view(rep.size(0), self.num_chunks, self.chunk_width)

    def forward(self, rep_egnn, rep_it):
        egnn_chunks = self._to_chunks(rep_egnn)
        it_chunks = self._to_chunks(rep_it)
        cross, _ = self.egnn_cross_attn(
            egnn_chunks,
            it_chunks,
            it_chunks,
            need_weights=False,
        )
        updated_egnn_chunks = egnn_chunks + self.fusion_gate * cross
        fused_chunks = updated_egnn_chunks.reshape(updated_egnn_chunks.size(0), -1)
        rep_fusion = self.fusion_norm(fused_chunks)
        return rep_fusion
