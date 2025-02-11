# Copyright 2020 DeepMind Technologies Limited.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Code related to resampling of weighted samples."""
from functools import partial
from typing import Tuple

import algorithms.common.types as tp
import chex
import jax
import jax.numpy as jnp

Array = tp.Array
RandomKey = tp.RandomKey
Samples = tp.Samples

assert_trees_all_equal_shapes = chex.assert_trees_all_equal_shapes


def log_effective_sample_size(log_weights: Array) -> Array:
    """Numerically stable computation of log of effective sample size.

    ESS := (sum_i weight_i)^2 / (sum_i weight_i^2) and so working in terms of logs
    log ESS = 2 log sum_i (log exp log weight_i) - log sum_i (exp 2 log weight_i )

    Args:
      log_weights: Array of shape (num_batch). log of normalized weights.
    Returns:
      Scalar log ESS.
    """
    chex.assert_rank(log_weights, 1)
    first_term = 2.0 * jax.scipy.special.logsumexp(log_weights)
    second_term = jax.scipy.special.logsumexp(2.0 * log_weights)
    chex.assert_equal_shape([first_term, second_term])
    return first_term - second_term


def simple_resampling(
    key: RandomKey, log_weights: Array, samples: Array  # Multinomial resampling?
) -> Tuple[Array, Array]:
    """Simple resampling of log_weights and samples pair.

    Randomly select possible samples with replacement proportionally to
    softmax(log_weights).

    Args:
      key: A Jax Random Key.
      log_weights: An array of size (num_batch,) containing the log weights.
      samples: An array of size (num_batch, num_dim) containing the samples.å
    Returns:
      New samples of shape (num_batch, num_dim) and weights of shape (num_batch,)
    """
    chex.assert_rank(log_weights, 1)
    num_batch = log_weights.shape[0]
    indices = jax.random.categorical(key, log_weights, shape=(num_batch,))
    take_lambda = lambda x: jnp.take(x, indices, axis=0)
    resamples = jax.tree_util.tree_map(take_lambda, samples)
    log_weights_new = -jnp.log(log_weights.shape[0]) * jnp.ones_like(log_weights)
    chex.assert_equal_shape([log_weights, log_weights_new])
    assert_trees_all_equal_shapes(resamples, samples)
    return resamples, log_weights_new


@jax.jit
def systematic_resampling(
    rng: RandomKey,
    log_weights: Array,
    samples: Array,
):
    r"""
    Copied from https://github.com/angusphillips/particle_denoising_diffusion_sampler/blob/main/pdds/resampling.py#L131
    Select elements from `samples` with weights defined by `log_weights` using systematic resampling.

    Parameters
    ----------
    rng:
        random key
    samples:
        a sequence of elements to be resampled. Must have the same length as `log_weights`

    Returns
    -------
    Dict object
        contains three attributes:
            * ``samples``, giving the resampled elements
            * ``lw``, giving the new logweights
            * ``resampled``, True
    """
    N = log_weights.shape[0]

    # permute order of samples
    rng, rng_ = jax.random.split(rng)
    log_weights = jax.random.permutation(rng_, log_weights)
    samples = jax.random.permutation(rng_, samples)

    # Generates the uniform variates depending on sampling mode
    rng, rng_ = jax.random.split(rng)
    sorted_uniform = (jax.random.uniform(rng_, (1,)) + jnp.arange(N)) / N

    # Performs resampling given the above uniform variates
    new_idx = jnp.searchsorted(
        jnp.cumsum(jax.scipy.special.logsumexp(log_weights)), sorted_uniform
    )
    samples = samples[new_idx, :]

    return samples, jnp.zeros(N) - jnp.log(N)


def selection_scheme(key, X, Y, args):
    """
    Apply a uniformly random chosen permutation to both arrays X and Y.

    Parameters:
    X (jax.numpy.ndarray): Array of shape (N, M).
    Y (jax.numpy.ndarray): Array of shape (N,).
    key (jax.random.PRNGKey): JAX random key.

    Returns:
    tuple: Permuted arrays (X_permuted, Y_permuted).
    """
    [use_sorting_scheme, proportion] = args
    N = X.shape[0]
    M = int(N * proportion)
    indices = None

    if not use_sorting_scheme:
        indices = jax.random.permutation(key, N)
    else:
        # Suppose we pick a size M subgroup to sort
        # We pick M to be the M/2 samples w/ lowest weights
        # and M/2 samples w/ the highest weights
        half_M = M // 2
        indices = jnp.argsort(Y)
        indices = jnp.concat((indices[-half_M:], indices[:-half_M]))

    X_permuted = X[indices]
    Y_permuted = Y[indices]
    return X_permuted, Y_permuted, M


def partial_resampling(
    key: RandomKey, log_weights: Array, samples: Array, args=[False, 0.5]
) -> Tuple[Array, Array]:
    """
    Resamples according to following algorithm:
    - select a subset of samples with prob p=resampling_proportion
    - apply multinomial resampling within this subset
    - Importantly: We can choose subsets not just randomly. We can choose subsets
                   based on logweights / location / however we like

    Args:
      key: A Jax Random Key.
      log_weights: An array of size (num_batch,) containing the log weights.
      samples: An array of size (num_batch, num_dim) containing the samples.
      args (static): [Whether or not to use pseudoclipping selection scheme, proportion to resample]
    Returns:
      New samples of shape (num_batch, num_dim) and weights of shape (num_batch,)

    """
    N = log_weights.shape[0]
    log_weights = jax.nn.log_softmax(log_weights)
    # permute order of samples
    key, gen1, gen2 = jax.random.split(key, 3)
    samples, log_weights, m_to_resample = selection_scheme(
        gen1, samples, log_weights, args
    )

    resampled_subset, subset_log_weights = simple_resampling(
        gen2, log_weights[:m_to_resample], samples[:m_to_resample]
    )
    subset_log_weights += jax.scipy.special.logsumexp(
        log_weights[:m_to_resample]
    )  # normalization constant
    samples = samples.at[:m_to_resample].set(resampled_subset)
    log_weights = log_weights.at[:m_to_resample].set(subset_log_weights)

    return samples, log_weights


def optionally_resample(
    key: RandomKey,
    log_weights: Array,
    samples: Samples,
    resample_threshold: Array,
    resampler,
) -> Tuple[Array, Array]:
    """Call simple_resampling on log_weights/samples if ESS is below threshold.

    The resample_threshold is interpretted as a fraction of the total number of
    samples. So for example a resample_threshold of 0.3 corresponds to an ESS of
    samples 0.3 * num_batch.

    Args:
      key: Jax Random Key.
      log_weights: Array of shape (num_batch,)
      samples: Array of shape (num_batch, num_dim)
      resample_threshold: scalar controlling fraction of total sample sized used.
    Returns:
      new samples of shape (num_batch, num_dim) and
    """
    lambda_no_resample = lambda x: (x[2], x[1])
    lambda_resample = lambda x: resampler(*x)
    threshold_sample_size = log_weights.shape[0] * resample_threshold
    log_ess = log_effective_sample_size(log_weights)
    return jax.lax.cond(
        log_ess < jnp.log(threshold_sample_size),
        lambda_resample,
        lambda_no_resample,
        (key, log_weights, samples),
    )


def get_resampler(identifier, resampler_args=None):
    if identifier == "multinomial":
        return simple_resampling
    elif identifier == "systematic":
        return systematic_resampling
    elif identifier == "partial_simple":
        return partial(partial_resampling, args=resampler_args)
    else:
        raise ValueError(f"No resampling scheme named {identifier}")
