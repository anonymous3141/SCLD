from functools import partial

import jax
import jax.numpy as jnp
from algorithms.common.eval_methods.utils import (avg_stddiv_across_marginals,
                                                  compute_reverse_ess,
                                                  moving_averages,
                                                  save_samples)
from eval import discrepancies


def get_eval_fn(rnd, target, target_samples, cfg):
    rnd_reverse = jax.jit(partial(rnd, prior_to_target=True))

    if cfg.compute_forward_metrics and target.can_sample:
        rnd_forward = jax.jit(partial(rnd, prior_to_target=False))

    logger = {
        "KL/elbo": [],
        "KL/eubo": [],
        "logZ/delta_forward": [],
        "logZ/forward": [],
        "logZ/delta_reverse": [],
        "logZ/reverse": [],
        "ESS/forward": [],
        "ESS/reverse": [],
        "discrepancies/mmd": [],
        "discrepancies/sd": [],
        "other/target_log_prob": [],
        "other/delta_mean_marginal_std": [],
        "other/EMC": [],
        "stats/step": [],
        "stats/wallclock": [],
        "stats/nfe": [],
        "log_var/log_var": [],
        "log_var/traj_bal_ln_z": [],
    }

    def short_eval(model_state, key):
        if isinstance(model_state, tuple):
            model_state1, model_state2 = model_state
            params = (model_state1.params, model_state2.params)
        else:
            params = (model_state.params,)
        samples, running_costs, stochastic_costs, terminal_costs = rnd_reverse(
            key, model_state, *params
        )[:4]

        log_is_weights = -(running_costs + stochastic_costs + terminal_costs)
        ln_z = jax.scipy.special.logsumexp(log_is_weights) - jnp.log(cfg.eval_samples)
        elbo = -jnp.mean(running_costs + terminal_costs)
        log_var = jnp.var(running_costs + terminal_costs, ddof=0)

        if target.log_Z is not None:
            logger["logZ/delta_reverse"].append(jnp.abs(ln_z - target.log_Z))

        logger["logZ/reverse"].append(ln_z)
        logger["KL/elbo"].append(elbo)
        logger["ESS/reverse"].append(
            compute_reverse_ess(log_is_weights, cfg.eval_samples)
        )
        logger["other/target_log_prob"].append(jnp.mean(target.log_prob(samples)))
        logger["other/delta_mean_marginal_std"].append(
            jnp.abs(avg_stddiv_across_marginals(samples) - target.marginal_std)
        )
        logger["log_var/log_var"].append(log_var)

        if cfg.compute_forward_metrics and target.can_sample:
            (
                fwd_samples,
                fwd_running_costs,
                fwd_stochastic_costs,
                fwd_terminal_costs,
            ) = rnd_forward(jax.random.PRNGKey(0), model_state, *params)[:4]
            fwd_log_is_weights = -(
                fwd_running_costs + fwd_stochastic_costs + fwd_terminal_costs
            )
            fwd_ln_z = -(
                jax.scipy.special.logsumexp(-fwd_log_is_weights)
                - jnp.log(cfg.eval_samples)
            )
            fwd_ess = jnp.exp(
                fwd_ln_z
                - (
                    jax.scipy.special.logsumexp(fwd_log_is_weights)
                    - jnp.log(cfg.eval_samples)
                )
            )
            eubo = jnp.mean(fwd_log_is_weights)

            if target.log_Z is not None:
                logger["logZ/delta_forward"].append(jnp.abs(fwd_ln_z - target.log_Z))
            logger["logZ/forward"].append(fwd_ln_z)
            logger["KL/eubo"].append(eubo)
            logger["ESS/forward"].append(fwd_ess)

        logger.update(target.visualise(samples=samples, show=cfg.visualize_samples))

        if cfg.compute_emc and cfg.target.has_entropy:
            logger["other/EMC"].append(target.entropy(samples))

        for d in cfg.discrepancies:
            logger[f"discrepancies/{d}"].append(
                getattr(discrepancies, f"compute_{d}")(target_samples, samples, cfg)
                if target_samples is not None
                else jnp.inf
            )

        if cfg.moving_average.use_ma:
            for key, value in moving_averages(
                logger, window_size=cfg.moving_average.window_size
            ).items():
                if isinstance(value, list):
                    value = value[0]
                if key in logger.keys():
                    logger[key].append(value)
                    logger[f"model_selection/{key}_MAX"].append(max(logger[key]))
                    logger[f"model_selection/{key}_MIN"].append(min(logger[key]))
                else:
                    logger[key] = [value]
                    logger[f"model_selection/{key}_MAX"] = [max(logger[key])]
                    logger[f"model_selection/{key}_MIN"] = [min(logger[key])]

        if cfg.save_samples:
            save_samples(cfg, logger, samples)

        return logger

    return short_eval, logger
