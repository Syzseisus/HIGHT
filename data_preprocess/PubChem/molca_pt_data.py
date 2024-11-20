import json
import os.path as osp

import torch
from tqdm import tqdm
from torch_geometric.data import InMemoryDataset


class PubChemDataset(InMemoryDataset):
    """
    x           (num_atoms, node_feat_dim)
    edge_index  (2, num_edges)
    edge_attr   (num_edges, edge_attr_dim)
    text        (str)
    smiles      (str)
    cid         (digit str)
    Ex)
        Data(x=[14, 9], edge_index=[2, 26], edge_attr=[26, 3], text='The molecule is an O-acylcarnitine having acetyl as the acyl substituent. It has a role as a human metabolite. It is functionally related to an acetic acid. It is a conjugate base of an O-acetylcarnitinium.
        The molecule is a natural product found in Pseudo-nitzschia multistriata, Euglena gracilis, and other organisms with data available.
        The molecule is a metabolite found in or produced by Saccharomyces cerevisiae. An acetic acid ester of CARNITINE that facilitates movement of ACETYL COA into the matrices of mammalian MITOCHONDRIA during the oxidation of FATTY ACIDS.',
        smiles='CC(=O)OC(CC(=O)[O-])C[N+](C)(C)C', cid='1')
    """

    def __init__(self, path):
        super(PubChemDataset, self).__init__()
        self.data, self.slices = torch.load(path)

    def __getitem__(self, idx):
        return self.get(idx)


if __name__ == "__main__":
    root = "/workspace/DATASET/MolCA/PubChem324kV2"

    whole_pair = {}
    ind = 0
    for split in ["pretrain", "train", "valid", "test"]:
        tmp_dataset = PubChemDataset(osp.join(root, f"{split}.pt"))
        tmp_pair = {}
        for i, data in enumerate(tqdm(tmp_dataset, total=len(tmp_dataset), desc=f"Processing {split} dataset...")):
            smiles = data.smiles
            text = data.text
            tmp_pair[i] = {"smiles": smiles, "text": text}
            whole_pair[ind] = {"smiles": smiles, "text": text}
        print(f"Save {split} dataset...")
        with open(osp.join(root, f"{split}.json"), "w") as f:
            json.dump(tmp_pair, f)
    print(f"Save whole dataset...")
    with open(osp.join(root, f"raw.json"), "w") as f:
        json.dump(whole_pair, f)
    print("DONE")
