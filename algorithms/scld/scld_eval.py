import os

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import wandb
from algorithms.common.eval_methods.utils import avg_stddiv_across_marginals
from algorithms.common.ipm_eval import discrepancies
from algorithms.scld.resampling import log_effective_sample_size
from jax.scipy.special import logsumexp


def plot_hist(data):
    plt.close()
    fig, ax = plt.subplots()
    ax.hist(data, bins=40)
    return {"figures/vis": [wandb.Image(fig)]}


def visualise_betas(params, beta_fn, num_steps):
    # visualise the annealing schedule if it is learnt
    plt.close()
    b = [beta_fn(i) for i in range(num_steps + 1)]
    fig, ax = plt.subplots()
    ax.plot(b)
    return {"figures/vis": [wandb.Image(fig)]}


def eval_scld(
    simulate, simulate_smc, get_schedule_and_prior_fn, target, target_samples, config
):

    # simulate fn rolls out without any smc
    # simulate_smc rolls out as dictated by the use_{resampling, markov}_inference config settings

    history = {
        "model_lnZ": [],
        "smc_lnZ": [],
        "model_ELBO": [],
        "smc_ELBO": [],
        "model_sd": [],
        "smc_sd": [],
        "ess_model": [],
        "delta_mean_marginal_std_model": [],
        "delta_mean_marginal_std_smc": [],
    }

    moving_averages = {
        "model_lnZ": [],
        "smc_lnZ": [],
        "model_ELBO": [],
        "smc_ELBO": [],
        "model_sd": [],
        "smc_sd": [],
        "ess_model": [],
        "delta_mean_marginal_std_model": [],
        "delta_mean_marginal_std_smc": [],
    }

    time_optimal_reached = {}

    def short_eval(model_state, params, key, it_num):
        """
        Evaluate function:
        Compute and log metrics of current sampler
        """
        logger = {}

        is_finished = it_num + 1 == config.algorithm.n_sim

        (
            model_samples_all,
            _,
            (model_lnz_est, model_elbo_est),
            (_, per_subtraj_rnds),
            subtrajs_model,
            log_ess_model,
        ) = simulate(key, model_state, params)
        (
            smc_samples_all,
            _,
            (smc_lnz_est, smc_elbo_est),
            (_, per_subtraj_rnds_smc),
            subtrajs_smc,
            log_ess_smc,
        ) = simulate_smc(key, model_state, params)
        total_rnds = per_subtraj_rnds.sum(axis=0)
        total_rnds_smc = per_subtraj_rnds_smc.sum(axis=0)

        model_samples = model_samples_all[-1]
        smc_samples = smc_samples_all[-1]
        assert model_samples.shape[0] == config.eval_samples

        _, initial_density, beta_fn, _ = get_schedule_and_prior_fn(params)

        if hasattr(config.algorithm, "prior"):
            dim = model_samples.shape[-1]
            for j in range(min(dim, 5)):
                logger[f"prior/dim_{j}_mean"] = params["params"]["prior_mean"][j]
                logger[f"prior/dim_{j}_std"] = jnp.exp(
                    params["params"]["prior_log_stds"][j]
                )
        logger[f"prior/max_diffusion"] = jnp.exp(params["params"]["log_max_diffusion"])
        logger["metric/model_lnZ"] = model_lnz_est
        logger["metric/smc_lnZ"] = smc_lnz_est
        logger["metric/model_ess"] = (
            jnp.exp(log_effective_sample_size(total_rnds)) / config.eval_samples
        )

        if target.log_Z is not None:
            logger["metric/model_delta_lnZ"] = jnp.abs(model_lnz_est - target.log_Z)
            logger["metric/smc_delta_lnZ"] = jnp.abs(smc_lnz_est - target.log_Z)

        logger["metric/model_ELBO"] = model_elbo_est
        logger["metric/smc_ELBO"] = smc_elbo_est
        logger["metric/model_target_llh"] = jnp.mean(target.log_prob(smc_samples))
        logger["metric/smc_target_llh"] = jnp.mean(target.log_prob(model_samples))

        if config.compute_emc and config.target.has_entropy:
            logger["metric/model_entropy"] = target.entropy(model_samples)
            logger["metric/smc_entropy"] = target.entropy(smc_samples)

        for d in config.discrepancies:

            logger[f"discrepancies/model_{d}"] = (
                getattr(discrepancies, f"compute_{d}")(
                    target_samples, model_samples, config
                )
                if target_samples is not None
                else jnp.inf
            )

            logger[f"discrepancies/smc_{d}"] = (
                getattr(discrepancies, f"compute_{d}")(
                    target_samples, smc_samples, config
                )
                if target_samples is not None
                else jnp.inf
            )

        logger["model_samples"] = target.visualise(
            model_samples, show=config.visualize_samples
        )
        logger["model_samples_smc"] = target.visualise(
            smc_samples, show=config.visualize_samples
        )

        if target_samples is not None and is_finished:
            logger["groundtruth_samples"] = target.visualise(
                target_samples, show=config.visualize_samples
            )

        logger["rnds/model_logrnds"] = plot_hist(total_rnds)
        logger["rnds/model_logrnds_smc"] = plot_hist(total_rnds_smc)

        if config.algorithm.annealing_schedule.schedule_type == "learnt":
            logger["other/learnt_annealing_schedule"] = visualise_betas(
                params, beta_fn, config.algorithm.num_steps
            )

        if config.algorithm.loss in ["rev_tb", "fwd_tb"]:
            logger["other/sum_log_Z_second_moment"] = jnp.sum(
                params["params"]["logZ_deltas"]
            )

        logger["metric/delta_mean_marginal_std_model"] = jnp.abs(
            avg_stddiv_across_marginals(model_samples) - target.marginal_std
        )
        logger["metric/delta_mean_marginal_std_smc"] = jnp.abs(
            avg_stddiv_across_marginals(smc_samples) - target.marginal_std
        )

        # code for tracking history of relevant values and computing metrics
        history["model_lnZ"].append(model_lnz_est)
        history["smc_lnZ"].append(smc_lnz_est)
        history["model_ELBO"].append(model_elbo_est)
        history["smc_ELBO"].append(smc_elbo_est)
        history["model_sd"].append(logger["discrepancies/model_sd"])
        history["smc_sd"].append(logger["discrepancies/smc_sd"])
        history["ess_model"].append(logger["metric/model_ess"])
        history["delta_mean_marginal_std_model"].append(
            logger["metric/delta_mean_marginal_std_model"]
        )
        history["delta_mean_marginal_std_smc"].append(
            logger["metric/delta_mean_marginal_std_smc"]
        )

        moving_average_width = (
            5
            if not hasattr(config.algorithm, "ma_length")
            else config.algorithm.ma_length
        )

        for key, value in history.items():
            logger[f"MovingAverages/{key}"] = sum(value[-moving_average_width:]) / len(
                value[-moving_average_width:]
            )
            moving_averages[key].append(logger[f"MovingAverages/{key}"])

        for key, array in moving_averages.items():
            # slightly inefficient
            if len(array) > moving_average_width:
                logger[f"model_selection/{key}_MovingAverage_MAX"] = max(
                    array[moving_average_width:]
                )
                logger[f"model_selection/{key}_MovingAverage_MIN"] = min(
                    array[moving_average_width:]
                )

                if (
                    logger[f"model_selection/{key}_MovingAverage_MAX"]
                    == array[moving_average_width:][-1]
                ):
                    time_optimal_reached[f"model_selection/{key}_MovingAverage_MAX"] = (
                        it_num
                    )
                if (
                    logger[f"model_selection/{key}_MovingAverage_MIN"]
                    == array[moving_average_width:][-1]
                ):
                    time_optimal_reached[f"model_selection/{key}_MovingAverage_MIN"] = (
                        it_num
                    )

        if is_finished:
            for key, value in time_optimal_reached.items():
                logger[f"Optimal_time_{key}"] = value

        return logger

    return short_eval
