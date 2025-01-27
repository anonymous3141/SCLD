from typing import List

import chex
import distrax
import jax
import jax.numpy as jnp
import jax.random as random
import matplotlib
import numpy as np
import numpyro.distributions as dist
import wandb
from matplotlib import pyplot as plt
from scipy.stats import wishart
from targets.base_target import Target
from utils.path_utils import project_path

# matplotlib.use('agg')


class GaussianMixtureModel(Target):
    def __init__(self, dim=3, num_components=15, seed=0):
        super().__init__(dim=dim, log_Z=0, can_sample=True)

        self.num_components = num_components

        # parameters
        min_mean_val = -10
        max_mean_val = 10
        min_val_mixture_weight = 0.3
        max_val_mixture_weight = 0.7
        degree_of_freedom_wishart = dim + 2

        seed = jax.random.PRNGKey(seed)

        # set mixture components
        locs = jax.random.uniform(
            seed, minval=min_mean_val, maxval=max_mean_val, shape=(num_components, dim)
        )
        covariances = []
        for _ in range(num_components):
            seed, subkey = random.split(seed)

            # Set the random seed for Scipy
            seed_value = random.randint(key=subkey, shape=(), minval=0, maxval=2**30)
            np.random.seed(seed_value)

            cov_matrix = wishart.rvs(df=degree_of_freedom_wishart, scale=jnp.eye(dim))
            covariances.append(cov_matrix)

        self.component_dist = distrax.MultivariateNormalFullCovariance(
            locs, jnp.array(covariances)
        )

        # set mixture weights
        uniform_mws = True
        if uniform_mws:
            mixture_weights = distrax.Categorical(
                logits=jnp.ones(num_components) / num_components
            )
        else:
            mixture_weights = distrax.Categorical(
                logits=dist.Uniform(
                    low=min_val_mixture_weight, high=max_val_mixture_weight
                ).sample(seed, sample_shape=(num_components,))
            )

        self.mixture_distribution = distrax.MixtureSameFamily(
            mixture_distribution=mixture_weights,
            components_distribution=self.component_dist,
        )

    def sample(self, seed: chex.PRNGKey, sample_shape: chex.Shape) -> chex.Array:
        return self.mixture_distribution.sample(seed=seed, sample_shape=sample_shape)

    def log_prob(self, x: chex.Array) -> chex.Array:
        batched = x.ndim == 2

        if not batched:
            x = x[None,]

        log_prob = self.mixture_distribution.log_prob(x)

        if not batched:
            log_prob = jnp.squeeze(log_prob, axis=0)

        return log_prob

    def entropy(self, samples: chex.Array = None):
        expanded = jnp.expand_dims(samples, axis=-2)
        # Compute `log_prob` in every component.
        idx = jnp.argmax(
            self.mixture_distribution.components_distribution.log_prob(expanded), 1
        )
        unique_elements, counts = jnp.unique(idx, return_counts=True)
        mode_dist = counts / samples.shape[0]
        entropy = -jnp.sum(
            mode_dist * (jnp.log(mode_dist) / jnp.log(self.num_components))
        )
        return entropy

    def visualise(
        self,
        samples: chex.Array = None,
        axes: List[plt.Axes] = None,
        show=False,
        clip=False,
    ) -> None:
        plt.clf()

        boarder = [-14, 9]
        # clipping samples because of FABs outlier
        if clip:
            samples = jnp.clip(samples, boarder[0], boarder[1])

        if self.dim == 2:
            fig = plt.figure()
            ax = fig.add_subplot()

            x, y = jnp.meshgrid(
                jnp.linspace(boarder[0], boarder[1], 100),
                jnp.linspace(boarder[0], boarder[1], 100),
            )
            grid = jnp.c_[x.ravel(), y.ravel()]
            pdf_values = jax.vmap(jnp.exp)(self.log_prob(grid))
            pdf_values = jnp.reshape(pdf_values, x.shape)
            # ax.contourf(x, y, pdf_values, levels=50)  # , cmap='viridis')
            ax.contour(x, y, pdf_values, levels=8)  # , cmap='viridis')
            if samples is not None:
                plt.scatter(
                    samples[:300, 0], samples[:300, 1], c="r", alpha=0.8, marker="x"
                )
            # plt.xlabel('X')
            # plt.ylabel('Y')
            # plt.colorbar()
            plt.xticks([])
            plt.yticks([])
            plt.xlim(boarder)
            plt.ylim(boarder)
            plt.axis("off")
            # plt.savefig(os.path.join(project_path('./figures/'), f"gmm2D.pdf"), bbox_inches='tight', pad_inches=0.1)

            try:
                wandb.log({"images/target_vis": wandb.Image(plt)})
            except:
                pass

            import os

            import tikzplotlib

            plt.savefig(
                os.path.join(project_path("./figures/"), f"gmm.pdf"),
                bbox_inches="tight",
                pad_inches=0.1,
            )
            tikzplotlib.save(os.path.join(project_path("./figures/"), f"gmm.tex"))

        else:
            target_samples = self.sample(jax.random.PRNGKey(0), (500,))
            plt.scatter(
                target_samples[:, 0], target_samples[:, 1], c="b", label="target"
            )
            plt.scatter(samples[:, 0], samples[:, 1], c="r", label="model")
            plt.legend()

            try:
                wandb.log({"images/target_vis": wandb.Image(plt)})
            except:
                pass

        if show:
            plt.show()

        plt.close()


if __name__ == "__main__":
    key = jax.random.PRNGKey(0)
    gmm = GaussianMixtureModel(dim=2, num_components=3, seed=7)
    samples = gmm.component_dist.sample(seed=key, sample_shape=(200,))
    avg_gaussian = jax.random.normal(key, (300, 2)) * 5 - 1.5
    sample = gmm.sample(key, (300,))
    print(sample)
    print(samples)
    print((gmm.log_prob(sample)).shape)
    print((jax.vmap(gmm.log_prob)(sample)).shape)
    gmm.visualise(samples=avg_gaussian.clip(min=-13.5, max=8.5), show=True)  # t1
    # gmm.visualise(samples=samples[:, 0].reshape(-1, 2), show=True)  # t2
    # gmm.visualise(samples=samples[:, :2].reshape(-1, 2), show=True)  # t3
    # gmm.visualise(samples=samples[:, :].reshape(-1, 2), show=True)  # t4

    # f_elbo = lambda x: -1 / x
    # f_eubo = lambda x, a, b: jnp.exp(-a * x) * jnp.sin(b * x) * 25 + 0.2
    # ln_z = lambda x: jnp.zeros_like(x)
    # x = jnp.linspace(0.1, 6, 100)
    # x_long = jnp.linspace(-0.1, 20, 100)
    # plt.plot(x, f_elbo(x), c='b')
    # plt.annotate(text='ELBO', xy=(5, -2), c='b')
    # plt.plot(x, f_eubo(x, 1, 0.5), c='r')
    # plt.annotate(text='EUBO', xy=(5, 2), c='r')
    # plt.plot(x_long, ln_z(x_long), c='k')
    # plt.xlim([0, 6])
    # plt.ylim([-5, 5])
    # plt.xticks([])
    # plt.yticks([])
    # plt.annotate(text='$\log Z$', xy=(6., 0.), c='k')
    # import tikzplotlib
    # import os
    # tikzplotlib.save(os.path.join(project_path('./figures/'), f"elbo_eubo.tex"))
    # plt.show()
