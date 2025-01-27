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

"""Code for variational inference (VI) with normalizing flows.

For background see:

Rezende and Mohamed. 2015. Variational Inference with Normalizing Flows.
International Conference of Machine Learning.

"""
import functools
from time import time

import algorithms.common.types as tp
import chex
import jax
import jax.numpy as jnp
import optax
import wandb
from algorithms.common.ipm_eval import discrepancies
from targets.base_target import Target
from utils.print_util import print_results

Array = jnp.ndarray
UpdateFn = tp.UpdateFn
OptState = tp.OptState
FlowParams = tp.FlowParams
FlowApply = tp.FlowApply
LogDensityNoStep = tp.LogDensityNoStep
InitialSampler = tp.InitialSampler
RandomKey = tp.RandomKey
assert_equal_shape = chex.assert_equal_shape
AlgoResultsTuple = tp.AlgoResultsTuple
ParticleState = tp.ParticleState

assert_trees_all_equal_shapes = chex.assert_trees_all_equal_shapes


def vi_free_energy(
    flow_params: FlowParams,
    key: RandomKey,
    initial_sampler: InitialSampler,
    initial_density: LogDensityNoStep,
    final_density: LogDensityNoStep,
    flow_apply: FlowApply,
    cfg,
):
    """The variational free energy used in VI with normalizing flows."""
    samples = initial_sampler(seed=key, sample_shape=(cfg.algorithm.batch_size,))
    transformed_samples, log_det_jacs = flow_apply(flow_params, samples)
    assert_trees_all_equal_shapes(transformed_samples, samples)
    log_density_target = final_density(transformed_samples)
    log_density_initial = initial_density(samples)
    assert_equal_shape([log_density_initial, log_density_target])
    log_density_approx = log_density_initial - log_det_jacs
    assert_equal_shape([log_density_approx, log_density_initial])
    free_energies = log_density_approx - log_density_target
    free_energy = jnp.mean(free_energies)
    return free_energy


def outer_loop_vi(
    initial_sampler: InitialSampler,
    opt_update: UpdateFn,
    opt_init_state: OptState,
    flow_init_params: FlowParams,
    flow_apply: FlowApply,
    flow_inverse_apply: FlowApply,
    initial_log_density: LogDensityNoStep,
    target: Target,
    cfg,
    save_checkpoint,
) -> AlgoResultsTuple:
    """The training loop for variational inference with normalizing flows.

    Args:
      initial_sampler: Produces samples from the base distribution.
      opt_update: Optax update function for the optimizer.
      opt_init_state: Optax initial state for the optimizer.
      flow_init_params: Initial params for the flow.
      flow_apply: A callable that evaluates the flow for given params and samples.
      key: A Jax random Key.
      initial_log_density: Function that evaluates the base density.
      target_log_density: Function that evaluates the target density.
      cfg: cfguration cfgDict.
      save_checkpoint: None or function that takes params and saves them.
    Returns:
      An AlgoResults tuple containing a summary of the results.
    """

    def eval_nfvi(
        key: RandomKey,
    ):
        """Estimate log normalizing constant using naive importance sampling."""
        samples = initial_sampler(seed=key, sample_shape=(cfg.eval_samples,))
        transformed_samples, log_det_jacs = flow_apply(flow_params, samples)
        assert_trees_all_equal_shapes(transformed_samples, samples)
        log_density_target = target_log_density(transformed_samples)
        log_density_initial = initial_log_density(samples)
        assert_equal_shape([log_density_initial, log_density_target])
        log_density_approx = log_density_initial - log_det_jacs
        log_ratio = log_density_target - log_density_approx
        ln_z = jax.scipy.special.logsumexp(jnp.array(log_ratio)) - jnp.log(
            cfg.eval_samples
        )
        elbo = jnp.mean(log_ratio)
        is_weights = jnp.exp(log_ratio)

        if target.log_Z is not None:
            logger["metric/delta_lnZ"] = jnp.abs(ln_z - target.log_Z)
        else:
            logger["metric/lnZ"] = ln_z
        logger["metric/ELBO"] = elbo
        logger["metric/reverse_ESS"] = jnp.sum(is_weights) ** 2 / (
            cfg.eval_samples * jnp.sum(is_weights**2)
        )
        logger["metric/target_llh"] = jnp.mean(target.log_prob(samples))

        if cfg.compute_emc:
            logger["metric/entropy"] = target.entropy(samples)

        for d in cfg.discrepancies:
            logger[f"discrepancies/{d}"] = (
                getattr(discrepancies, f"compute_{d}")(target_samples, samples, cfg)
                if target_samples is not None
                else jnp.inf
            )

        if cfg.compute_forward_metrics and (target_samples is not None):
            target_log_p = target_log_density(target_samples)
            prior_samples, inv_log_det_jacs = flow_inverse_apply(
                flow_params, target_samples
            )
            model_log_p = initial_log_density(prior_samples) + inv_log_det_jacs
            log_ratio = target_log_p - model_log_p
            eubo = jnp.mean(log_ratio)
            fwd_ln_z = -(
                jax.scipy.special.logsumexp(-log_ratio) - jnp.log(cfg.eval_samples)
            )
            fwd_ess = jnp.exp(
                fwd_ln_z
                - (jax.scipy.special.logsumexp(log_ratio) - jnp.log(cfg.eval_samples))
            )
            logger["metric/EUBO"] = eubo
            if target.log_Z is not None:
                logger["metric/delta_fwd_lnZ"] = jnp.abs(fwd_ln_z - target.log_Z)
            else:
                logger["metric/fwd_lnZ"] = fwd_ln_z
            logger["metric/rev_ESS"] = fwd_ess

        target.visualise(transformed_samples, show=cfg.visualize_samples)

    alg_cfg = cfg.algorithm
    eval_freq = alg_cfg.iters // cfg.n_evals

    vi_free_energy_short = functools.partial(
        vi_free_energy,
        initial_sampler=initial_sampler,
        initial_density=initial_log_density,
        final_density=target.log_prob,
        flow_apply=flow_apply,
        cfg=cfg,
    )

    free_energy_and_grad = jax.jit(jax.value_and_grad(vi_free_energy_short))

    flow_params = flow_init_params
    opt_state = opt_init_state

    @jax.jit
    def nfvi_update(curr_key, curr_flow_params, curr_opt_state):
        subkey, curr_key = jax.random.split(curr_key)
        new_free_energy, flow_grads = free_energy_and_grad(curr_flow_params, subkey)
        updates, new_opt_state = opt_update(flow_grads, curr_opt_state)
        new_flow_params = optax.apply_updates(curr_flow_params, updates)
        return curr_key, new_flow_params, new_free_energy, new_opt_state

    key = jax.random.PRNGKey(cfg.seed)

    target_log_density = target.log_prob
    target_samples = target.sample(jax.random.PRNGKey(0), (cfg.eval_samples,))

    test_elbos = []
    logger = {}
    timer = 0

    for step in range(alg_cfg.iters):
        with jax.profiler.StepTraceAnnotation("train", step_num=step):
            iter_time = time()
            key, flow_params, curr_free_energy, opt_state = nfvi_update(
                key, flow_params, opt_state
            )
            timer += time() - iter_time

            if step % eval_freq == 0:
                key, subkey = jax.random.split(key)
                eval_nfvi(subkey)
                test_elbos.append(logger["metric/ELBO"])
                logger["stats/step"] = step
                logger["stats/wallclock"] = timer
                logger["stats/nfe"] = (step + 1) * alg_cfg.batch_size

                print_results(step, logger, cfg)

                if cfg.use_wandb:
                    wandb.log(logger)
