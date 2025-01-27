"""
Code for Gaussian Mean Field Variational Inference (MFVI).
Straightforward minimization of reverse KL
Adapted from mfvi/mfvi_trainer.py
"""

from time import time

import distrax
import jax
import jax.numpy as jnp
import optax
from algorithms.common.utils import get_optimizer
from jax import grad, jit
from targets.base_target import Target


# Diagonal Gaussian Variational Distribution
def initialize_variational_params(dim, init_mean, init_diagonal_std):
    initial_mean = jnp.ones(dim) * init_mean
    initial_log_var = jnp.log(jnp.ones(dim) * init_diagonal_std) * 2
    return initial_mean, initial_log_var


# ELBO (Evidence Lower Bound) objective
def neg_elbo(params, key, target_log_density, num_samples):
    mean, log_var = params
    std = jnp.exp(0.5 * log_var)
    samples, log_q = distrax.MultivariateNormalDiag(mean, std).sample_and_log_prob(
        seed=key, sample_shape=(num_samples,)
    )
    log_p_x = jnp.mean(jax.vmap(target_log_density)(samples))
    elbo = log_p_x - jnp.mean(log_q)
    return -elbo


def sample(params, key, num_samples):
    mean, log_var = params
    std = jnp.exp(0.5 * log_var)
    return distrax.MultivariateNormalDiag(mean, std).sample(
        seed=key, sample_shape=(num_samples,)
    )


# Training loop with optax.adam
def mfvi_trainer(mfvi_cfg, seed, target: Target):

    init_mean = mfvi_cfg.init_mean
    init_std = mfvi_cfg.init_std
    num_its = mfvi_cfg.num_its
    dim = target.dim

    key = jax.random.PRNGKey(seed)
    params = initialize_variational_params(dim, init_mean, init_std)
    optimizer = get_optimizer(mfvi_cfg.step_size, None)
    opt_state = optimizer.init(params)
    target_log_density = target.log_prob

    @jit
    def update(params, opt_state, key):
        gradient = grad(neg_elbo)(params, key, target_log_density, mfvi_cfg.batch_size)
        updates, new_opt_state = optimizer.update(gradient, opt_state)
        new_params = optax.apply_updates(params, updates)
        return new_params, new_opt_state

    timer = 0
    for step in range(num_its):

        if step % 1000 == 0:
            print(f"MFVI step {step}")
        iter_time = time()
        key, subkey = jax.random.split(key)

        params, opt_state = update(params, opt_state, subkey)
        timer += time() - iter_time

    return params
