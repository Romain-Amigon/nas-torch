# -*- coding: utf-8 -*-
"""
Created on Sat Dec 13 00:12:46 2025

@author: Romain
"""


from dataclasses import dataclass
from typing import Type
import torch.nn as nn

@dataclass
class LinearCfg:
    in_features: int
    out_features: int
    activation: Type[nn.Module]

@dataclass
class Conv2dCfg:
    in_channels: int
    out_channels: int
    kernel_size: int | tuple
    stride: int = 1
    padding: int = 0
    activation: Type[nn.Module] = nn.ReLU

@dataclass
class DropoutCfg:
    p: float

@dataclass
class FlattenCfg:
    start_dim: int = 1

@dataclass
class MaxPool2dCfg:
    kernel_size: int | tuple
    stride: int | tuple | None = None  
    padding: int = 0
    dilation: int = 1
    ceil_mode: bool = False
    
@dataclass
class GlobalAvgPoolCfg:
    """
    Effectue une moyenne sur chaque canal (H, W) -> (1, 1).
    Transforme un tenseur (N, C, H, W) en (N, C), ce qui remplace le Flatten coûteux.
    """
    pass

@dataclass
class BatchNorm1dCfg:
    num_features: int 

@dataclass
class BatchNorm2dCfg:
    num_features: int 

@dataclass
class ResBlockCfg:
    """
    Représente un bloc résiduel : y = x + sub_layers(x)
    """
    sub_layers: list 
    use_projection: bool = False