import random
from copy import deepcopy

import torch
import lightning.pytorch as pl
from torch_geometric.data import Data
from torch.utils.data import DataLoader
from pytorch_lightning.utilities import CombinedLoader

from .molecule_dataset import MoleculeDataset

# atom + motif + graph
NUM_NODE_ATTR = 120
NUM_NODE_CHIRAL = 4
NUM_BOND_ATTR = 4


class MaskAtom:
    def __init__(self, num_atom_type, num_edge_type, mask_rate, mask_edge=False):
        """
        Randomly masks an atom, and optionally masks edges connecting to it.
        The mask atom type index is num_possible_atom_type
        The mask edge type index in num_possible_edge_type
        :param num_atom_type:
        :param num_edge_type:
        :param mask_rate: % of atoms to be masked
        :param mask_edge: If True, also mask the edges that connect to the
        masked atoms
        """
        self.num_atom_type = num_atom_type
        self.num_edge_type = num_edge_type
        self.mask_rate = mask_rate
        self.mask_edge = mask_edge

    def __call__(self, data, masked_atom_indices=None):
        """
        :param data: pytorch geometric data object. Assume that the edge
        ordering is the default pytorch geometric ordering, where the two
        directions of a single edge occur in pairs.
        Eg. data.edge_index = tensor([[0, 1, 1, 2, 2, 3],
                                     [1, 0, 2, 1, 3, 2]])
        :param masked_atom_indices: If None, then randomly samples num_atoms
        * mask rate number of atom indices
        Otherwise a list of atom idx that sets the atoms to be masked (for
        debugging only)
        :return: None, Creates new attributes in original data object:
        data.mask_node_idx
        data.mask_node_label
        data.mask_edge_idx
        data.mask_edge_label
        """

        if masked_atom_indices == None:
            # sample x distinct atoms to be masked, based on mask rate. But
            # will sample at least 1 atom
            num_atoms = data.x.size()[0]
            sample_size = int(num_atoms * self.mask_rate + 1)
            masked_atom_indices = random.sample(range(num_atoms), sample_size)

        # create mask node label by copying atom feature of mask atom
        mask_node_labels_list = []
        for atom_idx in masked_atom_indices:
            mask_node_labels_list.append(data.x[atom_idx].view(1, -1))
        data.mask_node_label = torch.cat(mask_node_labels_list, dim=0)
        data.masked_atom_indices = torch.tensor(masked_atom_indices)

        # modify the original node feature of the masked node
        for atom_idx in masked_atom_indices:
            data.x[atom_idx] = torch.tensor([self.num_atom_type, 0])

        if self.mask_edge:
            # create mask edge labels by copying edge features of edges that are bonded to
            # mask atoms
            connected_edge_indices = []
            for bond_idx, (u, v) in enumerate(data.edge_index.cpu().numpy().T):
                for atom_idx in masked_atom_indices:
                    if atom_idx in set((u, v)) and bond_idx not in connected_edge_indices:
                        connected_edge_indices.append(bond_idx)

            if len(connected_edge_indices) > 0:
                # create mask edge labels by copying bond features of the bonds connected to
                # the mask atoms
                mask_edge_labels_list = []
                for bond_idx in connected_edge_indices[::2]:  # because the
                    # edge ordering is such that two directions of a single
                    # edge occur in pairs, so to get the unique undirected
                    # edge indices, we take every 2nd edge index from list
                    mask_edge_labels_list.append(data.edge_attr[bond_idx].view(1, -1))

                data.mask_edge_label = torch.cat(mask_edge_labels_list, dim=0)
                # modify the original bond features of the bonds connected to the mask atoms
                for bond_idx in connected_edge_indices:
                    data.edge_attr[bond_idx] = torch.tensor([self.num_edge_type, 0])

                data.connected_edge_indices = torch.tensor(connected_edge_indices[::2])
            else:
                data.mask_edge_label = torch.empty((0, 2)).to(torch.int64)
                data.connected_edge_indices = torch.tensor(connected_edge_indices).to(torch.int64)

        return data

    def __repr__(self):
        return "{}(num_atom_type={}, num_edge_type={}, mask_rate={}, mask_edge={})".format(
            self.__class__.__name__, self.num_atom_type, self.num_edge_type, self.mask_rate, self.mask_edge
        )


class BatchMasking(Data):
    r"""A plain old python object modeling a batch of graphs as one big
    (dicconnected) graph. With :class:`torch_geometric.data.Data` being the
    base class, all its methods can also be used here.
    In addition, single graphs can be reconstructed via the assignment vector
    :obj:`batch`, which maps each node to its respective graph identifier.
    """

    def __init__(self, batch=None, **kwargs):
        super().__init__(**kwargs)
        self.batch = batch

    @staticmethod
    def from_data_list(data_list):
        r"""Constructs a batch object from a python list holding
        :class:`torch_geometric.data.Data` objects.
        The assignment vector :obj:`batch` is created on the fly."""
        keys = [set(data.keys()) for data in data_list]
        keys = list(set.union(*keys))
        assert "batch" not in keys

        batch = BatchMasking()

        for key in keys:
            batch[key] = []
        batch.batch = []

        cumsum_node = 0
        cumsum_edge = 0

        for i, data in enumerate(data_list):
            num_nodes = data.num_nodes
            batch.batch.append(torch.full((num_nodes,), i, dtype=torch.long))
            for key in data.keys():
                item = data[key]
                if key in ["edge_index", "masked_atom_indices"]:
                    item = item + cumsum_node
                elif key == "connected_edge_indices":
                    item = item + cumsum_edge
                batch[key].append(item)

            cumsum_node += num_nodes
            cumsum_edge += data.edge_index.shape[1]

        for key in keys:
            batch[key] = torch.cat(batch[key], dim=data_list[0].__cat_dim__(key, batch[key][0]))
        batch.batch = torch.cat(batch.batch, dim=-1)
        return batch.contiguous()

    def cumsum(self, key, item):
        r"""If :obj:`True`, the attribute :obj:`key` with content :obj:`item`
        should be added up cumulatively before concatenated together.
        .. note::
            This method is for internal use only, and should only be overridden
            if the batch concatenation process is corrupted for a specific data
            attribute.
        """
        return key in ["edge_index", "face", "masked_atom_indices", "connected_edge_indices"]

    @property
    def num_graphs(self):
        """Returns the number of graphs in the batch."""
        return self.batch[-1].item() + 1


class DataLoaderMaskingPred(DataLoader):
    r"""Data loader which merges data objects from a
    :class:`torch_geometric.data.dataset` to a mini-batch.
    Args:
        dataset (Dataset): The dataset from which to load the data.
        batch_size (int, optional): How may samples per batch to load.
            (default: :obj:`1`)
        shuffle (bool, optional): If set to :obj:`True`, the data will be
            reshuffled at every epoch (default: :obj:`True`)
    """

    def __init__(self, dataset, batch_size=1, shuffle=True, mask_rate=0.0, mask_edge=False, **kwargs):
        self._transform = MaskAtom(
            num_atom_type=NUM_NODE_ATTR,
            num_edge_type=NUM_BOND_ATTR,
            mask_rate=mask_rate,
            mask_edge=mask_edge,
        )
        super().__init__(dataset, batch_size, shuffle, collate_fn=self.collate_fn, **kwargs)

    def collate_fn(self, batch):
        transformed_batch = [self._transform(x) for x in batch]
        return BatchMasking.from_data_list(transformed_batch)


class MaskingDataModule(pl.LightningDataModule):
    def __init__(self, data_args, training_args):
        super().__init__()
        self.data_args = data_args
        self.training_args = training_args

    def prepare_data(self):
        super().prepare_data()
        pass

    def setup(self, stage=None):
        dataset = MoleculeDataset(self.data_args.data_path, self.data_args.dataset)
        self.dataset1 = dataset.shuffle()
        self.dataset2 = deepcopy(self.dataset1)
        self.dataset_org = deepcopy(self.dataset1)

    def _verify_loader_lengths(self, loaders):
        lengths = [len(loader) for loader in loaders.values()]
        if not all(x == lengths[0] for x in lengths):
            raise ValueError(
                f"All dataloaders must have the same length. "
                f"Got lengths: org={lengths[0]}, mask1={lengths[1]}, mask2={lengths[2]}"
            )

    def train_dataloader(self):
        kwargs = {
            "batch_size": self.training_args.per_device_train_batch_size // 3,  # 세 개니까
            "shuffle": False,
            "num_workers": self.training_args.num_workers,
            "pin_memory": True,
            "drop_last": True,
            "prefetch_factor": 4,
        }

        loader1 = DataLoaderMaskingPred(
            self.dataset1, mask_rate=self.data_args.mask_rate1, mask_edge=self.data_args.mask_edge, **kwargs
        )
        loader2 = DataLoaderMaskingPred(
            self.dataset2, mask_rate=self.data_args.mask_rate2, mask_edge=self.data_args.mask_edge, **kwargs
        )
        loader_org = DataLoaderMaskingPred(self.dataset_org, mask_rate=0.0, mask_edge=False, **kwargs)
        loaders = {"org": loader_org, "mask1": loader1, "mask2": loader2}
        self._verify_loader_lengths(loaders)

        return CombinedLoader(loaders, mode="max_size_cycle")

    def val_dataloader(self):
        return None
