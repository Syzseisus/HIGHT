import torch
import torch.nn as nn
import torch.nn.functional as F


class VectorQuantizer(nn.Module):
    """
    VQ-VAE layer: Input any tensor to be quantized.
    Args:
        embedding_dim (int): the dimensionality of the tensors in the
        quantized space. Inputs to the modules must be in this format as well.
        num_tokens (int): the number of vectors in the quantized space.
        beta (float): scalar which controls the weighting of the loss terms.
    """

    def __init__(self, embedding_dim, num_tokens, beta):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_tokens = num_tokens
        self.beta = beta

        assert num_tokens == 640

        # initialize embeddings
        self.embeddings = nn.Embedding(self.num_tokens, self.embedding_dim)

    def forward(self, x, e):
        """
        Args:
            x (n, 2) : atom indicator (atom_index, atom_chiral)
            e (n, d) : atom embedding from encdoer
        Return:
            quantized (n, d) : quantized atom embedding, feed to decoder
            loss () : quantization loss (last 2 term of Eq. 4)
        """
        encoding_indices = self.get_code_indices(x, e)  # x: B * H, encoding_indices: B
        quantized = self.quantize(encoding_indices)

        # embedding loss : move the embeddings towards the encoder's output aiming to update codebook
        codebook_loss = F.mse_loss(quantized, e.detach())

        # commitment loss : which encourages the output of the encoder to stay close to the chosen codebook embedding.
        gnn_loss = F.mse_loss(e, quantized.detach())

        # total vq loss
        vq_loss = codebook_loss + self.beta * gnn_loss

        # Straight Through Estimator
        quantized = e + (quantized - e).detach().contiguous()
        return quantized, {"codebook_vq_loss": codebook_loss, "gnn_vq_loss": gnn_loss, "total_vq_loss": vq_loss}

    def get_code_indices(self, x, e):
        # x: N * 2  e: N * E
        atom_type = x[:, 0]
        index_c = atom_type == 5  # 0 is for the extra mask tokens
        index_n = atom_type == 6
        index_o = atom_type == 7
        index_m = atom_type == 119
        index_g = atom_type == 120
        index_others = ~(index_c + index_n + index_o + index_m + index_g)
        index_list = [index_c, index_n, index_o, index_others, index_m, index_g]

        # compute L2 distance
        encoding_indices = torch.ones(x.size(0)).long().to(x.device)

        # divide the codebook into several groups
        # Sec 4.1, last paragraph:
        # | ~, the quantized codes of C, N and O are restricted in
        # | [1, 128], [129, 256], [257, 384], ... left rare atoms are restriced to [385, 512]
        # -> in code, [1, 378], [379, 434], [435, 489], and [490, 512]
        # -> 일단은 C, N, O, others 각각 128개씩, motif, graph 각각 64개 씩 -> 640차원
        end_points = [128, 256, 384, 512, 576, 640]

        start = 0
        for end, index in zip(end_points, index_list):
            distances = (
                torch.sum(e[index] ** 2, dim=1, keepdim=True)
                + torch.sum(self.embeddings.weight[start:end] ** 2, dim=1)
                - 2.0 * torch.matmul(e[index], self.embeddings.weight[start:end].t())
            )
            encoding_indices[index] = torch.argmin(distances, dim=1) + start
            start = end + 1

        return encoding_indices

    def quantize(self, encoding_indices):
        """Returns embedding tensor for a batch of indices."""
        return self.embeddings(encoding_indices)

    def from_pretrained(self, model_file):
        self.load_state_dict(torch.load(model_file))
