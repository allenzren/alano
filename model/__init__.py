import torch
from torch import nn
import numpy as np
from numpy import array
from torch.nn.utils import spectral_norm
from collections import OrderedDict

activation_dict = nn.ModuleDict({
    "ReLU": nn.ReLU(),
    "ELU": nn.ELU(),
    "Tanh": nn.Tanh(),
    "Identity": nn.Identity()
})