"""
Integrator implementing quadrature rule.
"""

from typing import Tuple
from typeguard import typechecked

from jaxtyping import Float, jaxtyped
import torch
from torch_nerf.src.renderer.integrators.integrator_base import IntegratorBase


class QuadratureIntegrator(IntegratorBase):
    """
    Numerical integrator which approximates integral using quadrature.
    """

    @jaxtyped
    @typechecked
    def integrate_along_rays(
        self,
        sigma: Float[torch.Tensor, "num_ray num_sample"],
        radiance: Float[torch.Tensor, "num_ray num_sample 3"],
        delta: Float[torch.Tensor, "num_ray num_sample"],
    ) -> Tuple[Float[torch.Tensor, "num_ray 3"], Float[torch.Tensor, "num_ray num_sample"]]:
        """
        Computes quadrature rule to approximate integral involving in volume rendering.
        Pixel colors are computed as weighted sums of radiance values collected along rays.

        For details on the quadrature rule, refer to 'Optical models for
        direct volume rendering (IEEE Transactions on Visualization and Computer Graphics 1995)'.

        Args:
            sigma: Density values sampled along rays.
            radiance: Radiance values sampled along rays.
            delta: Distance between adjacent samples along rays.

        Returns:
            rgbs: Pixel colors computed by evaluating the volume rendering equation.
            weights: Weights used to determine the contribution of each sample to the final pixel color.
                A weight at a sample point is defined as a product of transmittance and opacity,
                where opacity (alpha) is defined as 1 - exp(-sigma * delta).
        """
        # TODO
        # HINT: Look up the documentation of 'torch.cumsum'.
    
        # 1. Compute alpha (opacity) for each sample
        alpha = 1.0 - torch.exp(-sigma * delta)  # [N, M]

        # 2. Compute transmittance T_i = exp(-cumsum(sigma * delta))
        # Shifted cumulative sum for stable T_i computation
        accumulated = torch.cumsum(sigma * delta, dim=-1)  # [N, M]
        accumulated = torch.cat([torch.zeros_like(accumulated[:, :1]), accumulated[:, :-1]], dim=-1)
        transmittance = torch.exp(-accumulated)  # [N, M]

        # 3. Compute sample weights
        weights = transmittance * alpha  # [N, M]

        # 4. Weighted sum over radiance
        rgb = torch.sum(weights[..., None] * radiance, dim=1)  # [N, 3]

        return rgb, weights


        alpha = 1.0 - torch.exp(-sigma * delta)  # [N, M]
        accumulated = torch.cumsum(sigma * delta, dim=-1)
        accumulated = torch.cat([torch.zeros_like(accumulated[:, :1]), accumulated[:, :-1]], dim=-1)
        transmittance = torch.exp(-accumulated)
        weight = transmittance * alpha
        result = torch.sum(weight[..., None] * radiance, dim=1)
        return result.float(), weight.float()