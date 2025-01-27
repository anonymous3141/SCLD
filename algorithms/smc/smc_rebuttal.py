import time
from datetime import date

import jax

rng_key = jax.random.key(42)

import time
from typing import Callable, Tuple

import algorithms.common.types as tp
import chex
import distrax
import jax
import jax.numpy as jnp
import numpy as np
import wandb
from algorithms.common import flow_transport, resampling
from algorithms.common.eval_methods.sis_methods import get_eval_fn
from algorithms.common.eval_methods.utils import extract_last_entry
from algorithms.common.ipm_eval import discrepancies
from jax.scipy.stats import multivariate_normal
from targets.base_target import Target

import blackjax
# from blackjax.smc import extend_params
import blackjax.smc.base as base
import blackjax.smc.ess as ess
import blackjax.smc.resampling as blackjax_resampling
import blackjax.smc.solver as solver
import blackjax.smc.tempered as tempered
from blackjax import adaptive_tempered_smc, irmh
from blackjax.base import SamplingAlgorithm
from blackjax.smc.adaptive_tempered import build_kernel
from blackjax.smc.inner_kernel_tuning import \
    as_top_level_api as inner_kernel_tuning
from blackjax.smc.tuning.from_particles import (mass_matrix_from_particles,
                                                particles_covariance_matrix,
                                                particles_means,
                                                particles_stds)
from blackjax.types import ArrayLikeTree, PRNGKey

Array = tp.Array
LogDensityNoStep = tp.LogDensityNoStep
InitialSampler = tp.InitialSampler
RandomKey = tp.RandomKey
MarkovKernelApply = tp.MarkovKernelApply
LogDensityByStep = tp.LogDensityByStep
assert_equal_shape = chex.assert_equal_shape
assert_trees_all_equal_shapes = chex.assert_trees_all_equal_shapes
AlgoResultsTuple = tp.AlgoResultsTuple
ParticleState = tp.ParticleState

"""
Sorry in advance for the code quality
"""


def extend_params(params):
    """Given a dictionary of params, repeats them for every single particle. The expected
    usage is in cases where the aim is to repeat the same parameters for all chains within SMC.
    """

    return jax.tree.map(lambda x: jnp.asarray(x)[None, ...], params)


# adapted version of blackjax.annealed_and_tempered
def as_top_level_api(
    logprior_fn: Callable,
    loglikelihood_fn: Callable,
    mcmc_step_fn: Callable,
    mcmc_init_fn: Callable,
    resampling_fn: Callable,
    target_ess: float,
    root_solver: Callable = solver.dichotomy,
    num_mcmc_steps: int = 10,
    **extra_parameters,
) -> SamplingAlgorithm:
    """Implements the (basic) user interface for the Adaptive Tempered SMC kernel.

    Parameters
    ----------
    logprior_fn
        The log-prior function of the model we wish to draw samples from.
    loglikelihood_fn
        The log-likelihood function of the model we wish to draw samples from.
    mcmc_step_fn
        The MCMC step function used to update the particles.
    mcmc_init_fn
        The MCMC init function used to build a MCMC state from a particle position.
    mcmc_parameters
        The parameters of the MCMC step function.  Parameters with leading dimension
        length of 1 are shared amongst the particles.
    resampling_fn
        The function used to resample the particles.
    target_ess
        The number of effective sample size to aim for at each step.
    root_solver
        The solver used to adaptively compute the temperature given a target number
        of effective samples.
    num_mcmc_steps
        The number of times the MCMC kernel is applied to the particles per step.

    Returns
    -------
    A ``SamplingAlgorithm``.

    """
    kernel = build_kernel(
        logprior_fn,
        loglikelihood_fn,
        mcmc_step_fn,
        mcmc_init_fn,
        resampling_fn,
        target_ess,
        root_solver,
        **extra_parameters,
    )

    def init_fn(position, rng_key=None):
        del rng_key
        return tempered.init(position)

    def step_fn(rng_key: PRNGKey, state, mcmc_parameters):
        return kernel(
            rng_key,
            state,
            num_mcmc_steps,
            mcmc_parameters,
        )

    return SamplingAlgorithm(init_fn, step_fn)


def irmh_full_cov_experiment(
    dimensions,
    target_ess,
    n_particles,
    prior_log_prob,
    loglikelihood,
    prior_std,
    use_hmc,
    alg_cfg,
):
    num_mcmc_steps = 1 if use_hmc else 100
    kernel = irmh.build_kernel() if not use_hmc else blackjax.hmc.build_kernel()
    init_fn = irmh.init if not use_hmc else blackjax.hmc.init
    initial_parameter_value = {
        "means": jnp.zeros(dimensions)[None, :],
        "cov": jnp.eye(dimensions)[None, :] * (prior_std**2),
    }
    if use_hmc:
        initial_parameter_value = dict(
            step_size=jnp.array([alg_cfg.mcmc.hmc_step_times[0]]),
            inverse_mass_matrix=jnp.eye(dimensions)[None, :],
            num_integration_steps=jnp.array(
                [alg_cfg.mcmc.hmc_num_leapfrog_steps]
            ),  # 10
        )

    def step(key, state, logdensity, **extra_args):

        if not use_hmc:
            # irmh
            means = extra_args["means"]
            cov = extra_args["cov"]
            "We need step to be vmappable over the parameter space, so we wrap it to make all parameter Jax Arrays or JaxTrees"
            proposal_distribution = lambda key: jax.random.multivariate_normal(
                key, means, cov
            )

            def proposal_logdensity_fn(proposal, state):
                return jnp.log(
                    jax.scipy.stats.multivariate_normal.pdf(
                        state.position, mean=means, cov=cov
                    )
                )

        else:
            return kernel(key, state, logdensity, extra_args)

    def mcmc_parameter_update_fn(state, info):
        if not use_hmc:
            covariance = jnp.atleast_2d(particles_covariance_matrix(state.particles))
            return {
                "means": particles_means(state.particles)[None, :],
                "cov": covariance[None, :],
            }
        else:
            new_inv_mass_matrix = mass_matrix_from_particles(state.particles)
            return {"inverse_mass_matrix": new_inv_mass_matrix}

    kernel_tuned_proposal = inner_kernel_tuning(
        logprior_fn=prior_log_prob,
        loglikelihood_fn=loglikelihood,
        mcmc_step_fn=step,
        mcmc_init_fn=init_fn,
        resampling_fn=blackjax_resampling.systematic,
        smc_algorithm=adaptive_tempered_smc,
        mcmc_parameter_update_fn=mcmc_parameter_update_fn,
        # TODO: tune cov
        initial_parameter_value={
            "means": jnp.zeros(dimensions)[None, :],
            "cov": jnp.eye(dimensions)[None, :] * (prior_std**2),
        },
        target_ess=target_ess,
        num_mcmc_steps=num_mcmc_steps,
    )

    return kernel_tuned_proposal


def smc_inference_loop(rng_key, smc_kernel, initial_state, cfg):
    """Run the tempered SMC algorithm.

    We run the adaptive algorithm until the tempering parameter lambda reaches the value
    lambda=1.

    """
    inv_mass_matrix = jnp.eye(cfg.target.dim)

    # 1 hmc step with 10 leapfrog steps

    def step_schedule(lmbda):
        return jnp.interp(
            lmbda,
            jnp.array(cfg.algorithm.mcmc.hmc_step_times),
            jnp.array(cfg.algorithm.mcmc.hmc_step_sizes),
        )

    def get_hmc_parameters(lmbda, particles):

        if cfg.algorithm.mcmc.adaptive_tuning:
            return dict(
                step_size=jnp.array([step_schedule(lmbda)]),
                inverse_mass_matrix=mass_matrix_from_particles(particles)[None, :],
                num_integration_steps=jnp.array(
                    [cfg.algorithm.mcmc.hmc_num_leapfrog_steps]
                ),  # 10
            )
        else:
            return dict(
                step_size=jnp.array([step_schedule(lmbda)]),
                inverse_mass_matrix=inv_mass_matrix[None, :],
                num_integration_steps=jnp.array(
                    [cfg.algorithm.mcmc.hmc_num_leapfrog_steps]
                ),  # 10
            )

    def cond(carry):
        i, state, _k, _, _ = carry
        if cfg.algorithm.mcmc.mcmc_kernel == "hmc":
            return (state.lmbda < 1) & (i <= 5000)
        else:
            return (state.sampler_state.lmbda < 1) & (i <= 5000)

    @jax.jit
    def one_step(carry):
        i, state, k, lnZ_increment, elbo_increment = carry
        k, subk = jax.random.split(k, 2)

        if cfg.algorithm.mcmc.mcmc_kernel == "hmc":
            state, info = smc_kernel(
                subk, state, get_hmc_parameters(state.lmbda, state.particles)
            )
        else:
            state, info = smc_kernel(subk, state)

        return (
            i + 1,
            state,
            k,
            lnZ_increment + info.log_likelihood_increment,
            elbo_increment + info.elbo_increment,
        )

    n_iter, final_state, _, lnZ, elbo = jax.lax.while_loop(
        cond, one_step, (0, initial_state, rng_key, 0, 0)
    )

    return n_iter, final_state, lnZ, elbo


# the actual training loop
def smc_rebuttal(cfg, target):
    target_samples = target.sample(
        seed=jax.random.PRNGKey(cfg.seed), sample_shape=(cfg.eval_samples,)
    )
    final_log_density = target.log_prob
    dim = target.dim
    alg_cfg = cfg.algorithm
    mcmc_cfg = cfg.algorithm.mcmc
    n_samples = cfg.eval_samples
    key = jax.random.PRNGKey(cfg.seed)
    key, subkey = jax.random.split(key)

    initial_density = distrax.MultivariateNormalDiag(
        jnp.ones(dim) * alg_cfg.init_mean, jnp.ones(dim) * alg_cfg.init_std
    )
    log_density_initial = initial_density.log_prob
    initial_sampler = initial_density.sample

    def logV(x):
        # as required by blackjax
        return final_log_density(x) - log_density_initial(x)

    # tempered = blackjax.adaptive_tempered_smc(
    tempered = None

    if alg_cfg.mcmc.mcmc_kernel == "hmc":
        tempered = as_top_level_api(
            log_density_initial,
            logV,
            blackjax.hmc.build_kernel(),
            blackjax.hmc.init,
            # hmc_parameters, #extend_params(n_samples, hmc_parameters),
            blackjax_resampling.systematic,
            alg_cfg.target_ess,  # the target ESS level
            num_mcmc_steps=1,
        )
    else:
        tempered = irmh_full_cov_experiment(
            dim,
            alg_cfg.target_ess,
            n_samples,
            log_density_initial,
            logV,
            alg_cfg.init_std,
            alg_cfg.mcmc.mcmc_kernel == "hmc",
            alg_cfg,
        )

    rng_key, init_key, sample_key = jax.random.split(key, 3)
    initial_samples = initial_sampler(seed=init_key, sample_shape=(n_samples,))
    initial_smc_state = tempered.init(initial_samples)

    time_start = time.time()
    n_iter, smc_samples, lnZ, elbo = smc_inference_loop(
        sample_key, tempered.step, initial_smc_state, cfg
    )  # cfg.target.dim, step_schedule)
    time_elapsed = time.time() - time_start

    final_samples = (
        smc_samples.particles
        if alg_cfg.mcmc.mcmc_kernel == "hmc"
        else smc_samples.sampler_state.particles
    )

    logger = {}
    for d in cfg.discrepancies:
        if target_samples is not None:
            logger[f"discrepancies/{d}"] = getattr(discrepancies, f"compute_{d}")(
                target_samples, final_samples, cfg
            )
        else:
            logger[f"discrepancies/{d}"] = jnp.inf
        print(f"discrepancies/{d} = {logger[f'discrepancies/{d}']}")
    print("Number of steps in the adaptive algorithm: ", n_iter.item())
    print(f"lnZ={lnZ}, elbo={elbo}")

    logger["n_steps"] = n_iter.item()
    logger["lnZ"] = lnZ.item()
    logger["elbo"] = elbo.item()
    logger["wallclock"] = time_elapsed
    logger["vis"] = target.visualise(final_samples, show=True)
    wandb.log(logger)
