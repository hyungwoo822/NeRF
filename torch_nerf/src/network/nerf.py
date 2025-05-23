"""
Pytorch implementation of MLP used in NeRF (ECCV 2020).
"""

from typing import Tuple
from typeguard import typechecked

from jaxtyping import Float, jaxtyped
import torch
import torch.nn as nn


class NeRF(nn.Module):
    """
    A multi-layer perceptron (MLP) used for learning neural radiance fields.

    For architecture details, please refer to 'NeRF: Representing Scenes as
    Neural Radiance Fields for View Synthesis (ECCV 2020, Best paper honorable mention)'.

    Attributes:
        pos_dim (int): Dimensionality of coordinate vectors of sample points.
        view_dir_dim (int): Dimensionality of view direction vectors.
        feat_dim (int): Dimensionality of feature vector within forward propagation.
    """

    def __init__(
        self,
        pos_dim: int,
        view_dir_dim: int,
        feat_dim: int = 256,
    ) -> None:
        """
        Constructor of class 'NeRF'.
        """
        super().__init__()

        # TODO
        self.layers = nn.ModuleList()
        for i in range(8):
            if i == 4:
                self.layers.append(nn.Linear(feat_dim + pos_dim, feat_dim))
            elif i == 0:
                self.layers.append(nn.Linear(pos_dim, feat_dim))
            else:
                self.layers.append(nn.Linear(feat_dim, feat_dim))
        self.relu = nn.ReLU()

        # Density prediction head
        self.sigma_head = nn.Linear(feat_dim, 1)

        # Feature to color head
        self.feature_layer = nn.Linear(feat_dim, feat_dim)
        self.rgb_layer = nn.Sequential(
            nn.Linear(feat_dim + view_dir_dim, feat_dim // 2),
            nn.ReLU(),
            nn.Linear(feat_dim // 2, 3)
        )

    @jaxtyped
    @typechecked
    def forward(
        self,
        pos: Float[torch.Tensor, "num_sample pos_dim"],
        view_dir: Float[torch.Tensor, "num_sample view_dir_dim"],
    ) -> Tuple[Float[torch.Tensor, "num_sample 1"], Float[torch.Tensor, "num_sample 3"]]:
        # TODO
        x = pos
        for i in range(8):
            if i == 4:
                x = torch.cat([x, pos], dim=-1)
            x = self.relu(self.layers[i](x))

        sigma = self.sigma_head(x)  # Density prediction
        feat = self.feature_layer(x)

        h = torch.cat([feat, view_dir], dim=-1)
        rgb = self.rgb_layer(h)

        return sigma, rgb
