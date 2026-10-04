from itertools import product
from pathlib import Path
from typing import List, Tuple

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from lightgbm import LGBMClassifier
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator
from rdkit.Chem.Scaffolds import MurckoScaffold
from sklearn.metrics import roc_auc_score
from sklearn.svm import SVC, LinearSVC
from torch_geometric.datasets import MoleculeNet
from torch_geometric.loader import DataLoader
from tqdm import tqdm
