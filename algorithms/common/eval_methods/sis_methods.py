import jax.numpy as jnp
from algorithms.common.eval_methods.utils import (avg_stddiv_across_marginals,
                                                  moving_averages,
                                                  save_samples)
from algorithms.common.ipm_eval import discrepancies


def get_eval_fn(cfg, target, target_samples):
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
    }

    n_eval_samples = cfg.eval_samples

    def eval_fn(samples, elbo, rev_lnz, eubo, fwd_lnz):

        if target.log_Z is not None:
            logger["logZ/delta_reverse"].append(jnp.abs(rev_lnz - target.log_Z))

        logger["logZ/reverse"].append(rev_lnz)
        logger["KL/elbo"].append(elbo)
        logger["other/target_log_prob"].append(jnp.mean(target.log_prob(samples)))
        logger["other/delta_mean_marginal_std"].append(
            jnp.abs(avg_stddiv_across_marginals(samples) - target.marginal_std)
        )
        if cfg.compute_forward_metrics and (target_samples is not None):
            if target.log_Z is not None:
                logger["logZ/delta_forward"].append(jnp.abs(fwd_lnz - target.log_Z))
            logger["logZ/forward"].append(fwd_lnz)
            logger["KL/eubo"].append(eubo)

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

    return eval_fn, logger
