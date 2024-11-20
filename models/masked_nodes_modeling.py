import torch
import torch.nn as nn
from torch_geometric.nn import global_mean_pool

triplet_loss = nn.TripletMarginLoss(margin=0.0, p=2)


class MnM_Module(nn.Module):
    def __init__(self, gnn):
        super().__init__()
        self.gnn = gnn
        emb_dim = self.gnn.emb_dim
        self.pool = global_mean_pool
        self.projection_head = nn.Sequential(
            nn.Linear(emb_dim, emb_dim),
            nn.ReLU(inplace=True),
            nn.Linear(emb_dim, emb_dim),
        )

    def forward_cl(self, x, edge_index, edge_attr, batch):
        node_rep = self.gnn(x, edge_index, edge_attr)
        graph_rep = self.pool(node_rep, batch)
        graph_rep = self.projection_head(graph_rep)
        return node_rep, graph_rep

    def loss_cl(self, masked_graph_rep1, masked_graph_rep2):
        # L_con in Eq. 7
        T = 0.1
        batch_size, _ = masked_graph_rep1.size()
        masked_graph_rep1_norm = masked_graph_rep1.norm(dim=1)
        masked_graph_rep2_norm = masked_graph_rep2.norm(dim=1)

        numer = torch.einsum("ik,jk->ij", masked_graph_rep1, masked_graph_rep2)
        denom = torch.einsum("i,j->ij", masked_graph_rep1_norm, masked_graph_rep2_norm)
        sim_matrix = numer / denom
        sim_matrix = torch.exp(sim_matrix / T)
        pos_sim = sim_matrix[range(batch_size), range(batch_size)]
        loss = pos_sim / (sim_matrix.sum(dim=1) - pos_sim)
        loss = -torch.log(loss).mean()
        return loss

    def loss_tri(self, graph_rep, masked_graph_rep1, masked_graph_rep2):
        # L_tri in Eq.7
        loss = triplet_loss(graph_rep, masked_graph_rep1, masked_graph_rep2)
        return loss
