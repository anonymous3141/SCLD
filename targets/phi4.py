import math
from typing import List

import chex
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import wandb
from targets.base_target import Target


class Phi4Distr(Target):
    def __init__(
        self,
        dim: int = 4,  # we calculate the dim from the lat_shape
        kappa: float = 0.02,
        lambd: float = 0.022,
        # lat_shape: int = 2,
        sample_bounds=None,
        can_sample=True,
        log_Z=None,
        **kwargs,
    ):
        super().__init__(dim, log_Z, can_sample)
        # lat_shape = [int(jnp.sqrt(dim)), int(jnp.sqrt(dim))]
        lat_shape = [dim // 8, 8]
        self.kappa = kappa
        self.lambd = lambd
        prod = int(math.prod(lat_shape))
        dim = prod
        if len(lat_shape) != 2:
            raise ValueError(
                f"The lattice configuration has an invalid shape {len(lat_shape)} instead of 2.\n "
                "Only 2D systems are supported for the `Phi4Action` action."
            )
        if prod != dim:
            raise ValueError(
                f"The number of dimension {dim} does not match the desired lattice"
                f" shape {prod}. Please check and try again!"
            )
        self.lat_shape = lat_shape
        self.expectations = {}
        # reference vals obtained with HMC on small volumes 8x8
        if self.kappa == 0.2:
            self.expectations["absmag"] = 0.6239
        if self.kappa == 0.3:
            self.expectations["absmag"] = 2.0870
        if self.kappa == 0.5:
            self.expectations["absmag"] = 4.8298

    def log_prob(self, x: jnp.ndarray) -> jnp.ndarray:
        og_shape = x.shape
        x = x.reshape((-1, self.lat_shape[0], self.lat_shape[1]))

        kinetic = (
            (-2 * self.kappa) * x * (jnp.roll(x, 1, axis=-1) + jnp.roll(x, 1, axis=-2))
        )
        mass = (1 - 2 * self.lambd) * x**2
        inter = self.lambd * x**4

        action = (kinetic + mass + inter).reshape(og_shape)
        return -jnp.sum(action, axis=-1)

    def visualise(
        self, samples: chex.Array = None, axes=None, show=False, prefix=""
    ) -> dict:
        fig = plt.figure()
        ax = fig.add_subplot()
        magnetization = samples.mean(-1)
        ax.hist(magnetization, bins=20, edgecolor="black", density=True)
        plt.show()
        wb = {"figures/vis": [wandb.Image(fig)]}
        if show:
            plt.show()
        return wb

    def sample(self, seed: chex.PRNGKey, sample_shape: chex.Shape) -> chex.Array:
        return None
