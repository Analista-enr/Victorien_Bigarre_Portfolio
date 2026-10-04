import torch
from torch_geometric.datasets import MoleculeNet
from torch_geometric.loader import DataLoader
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold


def get_dataset(name: str):
    """
    :param name: name of the dataset
    """
    dataset = MoleculeNet(root="./data", name=name)
    return dataset


def clean_dataset(dataset):
    for x in dataset:
        nb_nan = torch.isnan(x.y).sum()
        if nb_nan > 0:
            dataset.remove(x)
    return dataset
