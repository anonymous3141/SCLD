import os
from typing import List

import chex
import distrax
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import wandb
from targets.base_target import Target
from utils.path_utils import project_path


def visualize_samples(samples, num_goals, show=False):
    def visualize_n_link(theta, num_dimensions, l):
        x = [0]
        y = [0]
        for i in range(0, num_dimensions):
            y.append(y[-1] + l[i] * np.sin(np.sum(theta[: i + 1])))
            x.append(x[-1] + l[i] * np.cos(np.sum(theta[: i + 1])))
            ax.plot(
                [x[-2], x[-1]],
                [y[-2], y[-1]],
                color="k",
                linestyle="-",
                linewidth=2,
                alpha=0.3,
            )
        ax.plot(x[-1], y[-1], "o", c="k")
        return ax

    fig, ax = plt.subplots()
    num_dimensions = samples.shape[1]

    if num_goals == 1:
        ax.plot(0.7 * num_dimensions, 0, "rx")
        ax.set_xlim([-0.2 * num_dimensions, num_dimensions])
        ax.set_ylim([-0.5 * num_dimensions, 0.5 * num_dimensions])
    else:
        mx = [1, 0, -1, 0]
        my = [0, 1, 0, -1]
        for i in range(4):
            ax.plot(0.7 * num_dimensions * mx[i], 0.7 * num_dimensions * my[i], "rx")
        ax.set_xlim([-num_dimensions, num_dimensions])
        ax.set_ylim([-num_dimensions, num_dimensions])

    [num_samples, num_dimensions] = samples.shape
    for i in range(0, num_samples):
        visualize_n_link(samples[i], num_dimensions, np.ones(num_dimensions))

    wb = {"figures/vis": [wandb.Image(fig)]}
    if show:
        plt.show()
    return wb

    # import tikzplotlib
    # tikzplotlib.save(os.path.join(project_path('./figures/'), f"robot.tex"))


def sample_without_replacement(
    key: chex.Array, logits: chex.Array, n: int
) -> chex.Array:
    # https://timvieira.github.io/blog/post/2014/07/31/gumbel-max-trick/
    key1, key2 = jax.random.split(key)
    z = jax.random.gumbel(key=key1, shape=logits.shape)
    # vals, indices = jax.lax.approx_max_k(z + logits, n)
    vals, indices = jax.lax.top_k(z + logits, n)
    indices = jax.random.permutation(key2, indices)
    return indices


class PlanarRobot(Target):
    def __init__(
        self,
        dim: int,
        num_goals: int = 1,
        prior_std=2e-1,
        likelihood_std=1e-2,
        log_Z=None,
        can_sample=False,
        sample_bounds=None,
    ):
        super().__init__(dim, log_Z, can_sample)
        self.num_links = dim
        prior_stds = jnp.full((dim,), prior_std)
        prior_stds = prior_stds.at[0].set(1.0)
        self.prior = distrax.MultivariateNormalDiag(
            loc=jnp.zeros(dim), scale_diag=prior_stds
        )
        self.link_lengths = jnp.ones(self.dim)

        # Load ground truth samples (for 1 goal). gt_samples is (num_samples = 10k, 10)
        self.gt_samples = jnp.array(
            jnp.load(project_path("targets/data/planar_robot_gt_10k.npz"))["arr_0"]
        )
        self.num_gt_samples = self.gt_samples.shape[0]

        if num_goals == 1:
            self.goals = jnp.array([[7.0, 0.0]], dtype=jnp.float32)
        elif num_goals == 4:
            self.goals = jnp.array(
                [[7.0, 0.0], [-7.0, 0.0], [0.0, 7.0], [0.0, -7.0]], dtype=jnp.float32
            )
        else:
            raise ValueError("Number of goals must be 1 or 4")

        self.goal_Gaussians = []
        for goal in self.goals:
            goal_std = jnp.ones(2) * likelihood_std
            self.goal_Gaussians.append(
                distrax.MultivariateNormalDiag(loc=goal, scale_diag=goal_std)
            )

        if num_goals == 4:
            # Works by symmetry argument hopefully
            rotations = [
                self.gt_samples.at[:, 0].add(i * jnp.pi / 2) for i in range(-2, 2)
            ]
            sample_and_rotations = jnp.concat(rotations)  # (num_samples * 4, 10)
            sample_and_rotations = sample_and_rotations.at[:, 0].set(
                jnp.mod(sample_and_rotations[:, 0], 2 * jnp.pi) - jnp.pi
            )

            log_probs_unrotated = self.log_prob(self.gt_samples)
            log_rnds = self.log_prob(sample_and_rotations) - jnp.tile(
                log_probs_unrotated, 4
            )  # rnds

            # Gumbel-Softmax!
            gt_sample_indices = sample_without_replacement(
                jax.random.PRNGKey(0), log_rnds, self.num_gt_samples
            )
            self.gt_samples = sample_and_rotations[gt_sample_indices]

    def likelihood(self, pos):
        likelihoods = jnp.stack(
            [goal.log_prob(pos) for goal in self.goal_Gaussians], axis=0
        )
        return jnp.max(likelihoods, axis=0)

    def forward_kinematics(
        self, theta
    ):  # todo implement the batched version from oleg and follow the other target functions
        y = jnp.zeros(theta.shape[0])
        x = jnp.zeros(theta.shape[0])
        for i in range(self.dim):
            y += self.link_lengths[i] * jnp.sin(jnp.sum(theta[:, : i + 1], axis=1))
            x += self.link_lengths[i] * jnp.cos(jnp.sum(theta[:, : i + 1], axis=1))
        return jnp.stack((x, y), axis=1)

    def log_prob(self, theta):
        batched = theta.ndim == 2

        if not batched:
            theta = theta[None,]

        log_prob = self.prior.log_prob(theta) + self.likelihood(
            self.forward_kinematics(theta)
        )

        if not batched:
            log_prob = jnp.squeeze(log_prob, axis=0)

        return log_prob

    @property
    def marginal_std(self):
        # numerical integration
        from algorithms.common.eval_methods.utils import \
            avg_stddiv_across_marginals

        samples = self.sample(jax.random.PRNGKey(0), (2000,))
        return avg_stddiv_across_marginals(samples)

    def visualise(
        self, samples: chex.Array = None, axes=None, show=False, prefix=""
    ) -> dict:
        """Visualise samples from the model."""
        plt.close()
        num_samples = 1000
        idx = jax.random.choice(
            jax.random.PRNGKey(0), samples.shape[0], shape=(num_samples,)
        )
        return visualize_samples(samples[idx], len(self.goals), show=show)

    def sample(self, seed: chex.PRNGKey, sample_shape: chex.Shape) -> chex.Array:
        # Generate a subset of the ground truth sample set

        indices = jax.random.choice(
            seed, self.num_gt_samples, shape=sample_shape, replace=False
        )
        # Use the generated indices to select the subset

        return self.gt_samples[indices]


if __name__ == "__main__":
    pr = PlanarRobot(dim=10)
    samples = pr.sample(jax.random.PRNGKey(0), (2000,))
    pr.visualise(samples, show=True)
