import torch.nn as nn
from EGNN import build_egnn as build_residue_egnn
from Interatomic import build_it_encoder
from Fusion import CrossAttentionFusion

class InteractionHead(nn.Module):
    def __init__(self, rep_dim):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(rep_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.0),
            nn.Linear(256, 1),
        )

    def forward(self, rep):
        return self.mlp(rep).squeeze(-1)

class AffinityPredictor(nn.Module):
    def __init__(self, egnn_encoder, it_encoder, egnn_dim):
        super().__init__()
        self.egnn_encoder = egnn_encoder
        self.it_encoder = it_encoder
        self.fusion = CrossAttentionFusion(egnn_dim=egnn_dim)
        self.head = InteractionHead(rep_dim=self.fusion.fusion_dim)

    def _encode_egnn(self, residue_graph_batch):
        return self.egnn_encoder(
            residue_graph_batch.x,
            residue_graph_batch.pos,
            residue_graph_batch.edge_index,
            residue_graph_batch.batch,
            cluster=getattr(residue_graph_batch, "cluster", None),
            num_graphs=residue_graph_batch.num_graphs,
        )

    def _encode_it(self, token_batch):
        return self.it_encoder(
            token_batch.x,
            pos=token_batch.pos,
            batch=token_batch.batch,
            segment=getattr(token_batch, "segment", None),
        )

    def forward(self, residue_graph_batch=None, token_batch=None):
        rep_egnn = self._encode_egnn(residue_graph_batch)
        rep_it = self._encode_it(token_batch)
        rep_fusion = self.fusion(rep_egnn, rep_it)
        return self.head(rep_fusion)

class SingleBranchPredictor(nn.Module):
    def __init__(self, encoder, rep_dim, branch):
        super().__init__()
        self.branch = str(branch)
        if self.branch == "egnn":
            self.egnn_encoder = encoder
        else:
            self.it_encoder = encoder
        self.head = InteractionHead(rep_dim=rep_dim)

    def _encode_egnn(self, residue_graph_batch):
        return self.egnn_encoder(
            residue_graph_batch.x,
            residue_graph_batch.pos,
            residue_graph_batch.edge_index,
            residue_graph_batch.batch,
            cluster=getattr(residue_graph_batch, "cluster", None),
            num_graphs=residue_graph_batch.num_graphs,
        )

    def _encode_it(self, token_batch):
        return self.it_encoder(
            token_batch.x,
            pos=token_batch.pos,
            batch=token_batch.batch,
            segment=getattr(token_batch, "segment", None),
        )

    def forward(self, residue_graph_batch=None, token_batch=None):
        if self.branch == "egnn":
            rep_egnn = self._encode_egnn(residue_graph_batch)
            return self.head(rep_egnn)
        rep_it = self._encode_it(token_batch)
        return self.head(rep_it)

def _build_model(train_params, residue_in_dim, token_in_dim):
    egnn_encoder = (
        build_residue_egnn(
            in_dim=residue_in_dim,
        )
        if train_params.USE_EGNN_BRANCH
        else None
    )
    it_encoder = (
        build_it_encoder(in_dim=token_in_dim)
        if train_params.USE_IT_BRANCH
        else None
    )
    if train_params.USE_EGNN_BRANCH and train_params.USE_IT_BRANCH:
        model = AffinityPredictor(
            egnn_encoder=egnn_encoder,
            it_encoder=it_encoder,
            egnn_dim=egnn_encoder.out_dim,
        )
    elif train_params.USE_EGNN_BRANCH:
        model = SingleBranchPredictor(
            encoder=egnn_encoder,
            rep_dim=egnn_encoder.out_dim,
            branch="egnn",
        )
    elif train_params.USE_IT_BRANCH:
        model = SingleBranchPredictor(
            encoder=it_encoder,
            rep_dim=it_encoder.out_dim,
            branch="it",
        )
    model = model.to(train_params.DEVICE)
    return model
