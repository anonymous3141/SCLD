import algorithms.common.types as tp
import chex
import jax
import jax.numpy as jnp
import numpyro.distributions as npdist
import optax
from flax import traverse_util
from flax.traverse_util import flatten_dict, unflatten_dict

Array = tp.Array
FlowApply = tp.FlowApply
FlowParams = tp.FlowParams
LogDensityByStep = tp.LogDensityByStep
LogDensityNoStep = tp.LogDensityNoStep
MarkovKernelApply = tp.MarkovKernelApply
AcceptanceTuple = tp.AcceptanceTuple
RandomKey = tp.RandomKey
Samples = tp.Samples
assert_equal_shape = chex.assert_equal_shape
assert_trees_all_equal_shapes = chex.assert_trees_all_equal_shapes


class GeometricAnnealingSchedule(object):
    """Container containing the info to build geometric annealing schedules."""

    def __init__(
        self,
        initial_log_density: LogDensityNoStep,
        final_log_density: LogDensityNoStep,
        num_temps: int,
        target_clip=None,
        schedule_type: str = "uniform",
    ):
        self._initial_log_density = initial_log_density
        if target_clip > 0.0:
            self._final_log_density = lambda x: jnp.clip(
                final_log_density(x), -target_clip, target_clip
            )
        else:
            self._final_log_density = final_log_density
        self._num_temps = num_temps
        self.schedule_type = schedule_type


def reshape_to_match_num_dims(Y, X):
    # Number of dimensions to add
    num_dims_to_add = X.ndim - 1
    new_shape = Y.shape + (1,) * num_dims_to_add
    Y_reshaped = Y.reshape(new_shape)
    return Y_reshaped


def gradient_step(model_state, grads):
    grads_flat = flatten_dict(grads)
    grads_avg = unflatten_dict(jax.tree_map(lambda g: g.mean(0), grads_flat))
    return model_state.apply_gradients(grads=grads_avg)


def sample_kernel(rng_key, mean, scale):
    eps = jax.random.normal(rng_key, shape=(mean.shape[0],))
    return mean + scale * eps


def log_prob_kernel(x, mean, scale):
    dist = npdist.Independent(npdist.Normal(loc=mean, scale=scale), 1)
    return dist.log_prob(x)


def print_results(step, logger, config):
    if config.verbose:
        try:
            print(
                f'Step {step}: ELBO w. SMC {logger["metric/smc_ELBO"]}; ΔlnZ {logger["metric/smc_delta_lnZ"]}'
            )
            print(
                f'Step {step}: ELBO w/o. SMC {logger["metric/model_ELBO"]}; ΔlnZ {logger["metric/model_delta_lnZ"]}'
            )
        except:
            print(f'Step {step}: ELBO w. SMC {logger["metric/smc_ELBO"]}')
            print(f'Step {step}: ELBO w/o. SMC {logger["metric/model_ELBO"]}')


def flattened_traversal(fn):
    def mask(data):
        flat = traverse_util.flatten_dict(data)
        return traverse_util.unflatten_dict({k: fn(k, v) for k, v in flat.items()})

    return mask


def make_lr_scheduler(cfg):
    if (cfg.use_warmup is False) and (cfg.use_decay is False):
        return optax.constant_schedule(cfg.step_size)

    elif cfg.use_warmup is False:
        return optax.exponential_decay(
            init_value=cfg.step_size,
            transition_steps=1000,
            decay_rate=cfg.decay_factor_per_thousand,
            transition_begin=cfg.num_steps_before_start_decay,
            end_value=cfg.final_lr,
        )

    elif cfg.use_decay is False:
        return optax.warmup_exponential_decay_schedule(
            init_value=cfg.initial_lr,
            peak_value=cfg.step_size,
            warmup_steps=cfg.num_warmup_steps,
            transition_steps=1000,
            decay_rate=1,
            transition_begin=0,
            end_value=cfg.step_size,
        )
    else:
        return optax.warmup_exponential_decay_schedule(
            init_value=cfg.initial_lr,
            peak_value=cfg.step_size,
            warmup_steps=cfg.num_warmup_steps,
            transition_steps=1000,
            decay_rate=cfg.decay_factor_per_thousand,
            transition_begin=cfg.num_steps_before_start_decay,
            end_value=cfg.final_lr,
        )
