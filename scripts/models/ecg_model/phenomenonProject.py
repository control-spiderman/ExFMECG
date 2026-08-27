"""Projection heads for the ExFMECG concept space."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class PhenomenonProjector(nn.Module):
    def __init__(
        self,
        in_ch=512,
        mid_ch=128,
        t_pool=4,
        n_concepts=570,
        cat_num_classes=None,
        n_binary=None,
    ):
        super().__init__()
        self.in_ch = in_ch
        self.mid_ch = mid_ch
        self.t_pool = t_pool
        self.n_concepts = n_concepts
        feature_dim = mid_ch * (256 // t_pool)

        self.head_binary = nn.Linear(
            feature_dim,
            n_binary if n_binary is not None else n_concepts,
        )
        nn.init.xavier_uniform_(self.head_binary.weight)
        nn.init.zeros_(self.head_binary.bias)
        self.head = self.head_binary

        if cat_num_classes is None:
            self.head_regression = None
            self.head_categorical = None
        else:
            self.head_regression = nn.Linear(feature_dim, n_concepts)
            nn.init.xavier_uniform_(self.head_regression.weight)
            nn.init.zeros_(self.head_regression.bias)
            self.head_categorical = nn.ModuleList(
                nn.Linear(feature_dim, int(classes))
                for classes in cat_num_classes
            )

    def forward(self, features):
        grid = F.avg_pool2d(features, kernel_size=4, stride=4)
        return self.head(grid.flatten(1)), grid

    @torch.no_grad()
    def concept_from_cam(self, features, temporal_attention):
        if temporal_attention.ndim != 3 or temporal_attention.shape[1] != 1:
            raise ValueError("temporal_attention must have shape (B, 1, 256)")
        weighted = features * temporal_attention
        grid = F.avg_pool2d(weighted, kernel_size=4, stride=4)
        return self.head(grid.flatten(1))
