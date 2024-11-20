import os
import os.path as osp
from itertools import repeat
import torch
from torch_geometric.data import Data, InMemoryDataset, Batch
import lightning.pytorch as pl
from torch.utils.data import DataLoader


class MoleculeDataset(InMemoryDataset):
    def __init__(self, data_path, dataset, empty=False) -> None:
        self.root = osp.join(data_path, dataset)

        super().__init__(root=self.root)

        if not empty:
            self._data, self.slices = torch.load(self.processed_paths[0])

    def get(self, idx):
        data = Data()
        for key in self._data.keys():
            item, slices = self._data[key], self.slices[key]
            s = list(repeat(slice(None), item.dim()))
            s[data.__cat_dim__(key, item)] = slice(slices[idx], slices[idx + 1])
            data[key] = item[s]

        return data

    @property
    def raw_file_names(self):
        file_name_list = os.listdir(self.raw_dir)
        return file_name_list

    @property
    def processed_file_names(self):
        return "geometric_data_processed.pt"


class MoleculeDataModule(pl.LightningDataModule):
    def __init__(self, data_args, training_args):
        super().__init__()
        self.data_args = data_args
        self.training_args = training_args

    def setup(self, stage=None):
        self.train_dataset = MoleculeDataset(self.data_args.data_path, self.data_args.dataset)

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.training_args.per_device_train_batch_size,
            shuffle=True,
            num_workers=self.training_args.num_workers,
            collate_fn=Batch.from_data_list,
            pin_memory=True,
            drop_last=True,
            prefetch_factor=4,
        )
