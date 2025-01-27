import jax
import jax.numpy as jnp
from algorithms.scld.is_weights import (per_sample_sub_traj_is_weight,
                                        per_subtraj_log_is)


### Update functions for For TB-Loss ###
def lnZ_update_vanilla(old_lnZ, log_ws, lr):
    return (1 - 2 * lr) * old_lnZ + 2 * lr * log_ws.mean()


def lnZ_update_jensen(old_lnZ, log_ws, lr, batch_size):
    # equivalent to Z_new = (1-2a)Z_old + 2a * mean(rnds)
    # except we work in logSpace. This is preliminarily tested
    return jax.scipy.special.logsumexp(
        jnp.append(log_ws + jnp.log(2 * lr / batch_size), old_lnZ + jnp.log(1 - 2 * lr))
    )


### All Losses ###


def sub_traj_fwd_kl(
    keys,
    samples,
    samples_next,
    model_state,
    params,
    sim_tuple,
    traj_start,
    traj_end,
    traj_idx,
    traj_length,
    forward=False,
    stop_grad=False,
    per_sample_rnd_fn=per_sample_sub_traj_is_weight,
):
    sub_traj = traj_start, traj_end, traj_idx, traj_length
    w, samples_new = jax.vmap(
        per_sample_rnd_fn, in_axes=(0, 0, None, None, None, None, None, None)
    )(keys, samples_next, model_state, params, sim_tuple, sub_traj, forward, stop_grad)
    return w.mean(), (w, samples_new)


def sub_traj_rev_kl(
    keys,
    samples,
    samples_next,
    model_state,
    params,
    sim_tuple,
    traj_start,
    traj_end,
    traj_idx,
    traj_length,
    forward=True,
    stop_grad=False,
    per_sample_rnd_fn=per_sample_sub_traj_is_weight,
):
    sub_traj = traj_start, traj_end, traj_idx, traj_length
    w, samples_new = jax.vmap(
        per_sample_rnd_fn, in_axes=(0, 0, None, None, None, None, None, None)
    )(keys, samples, model_state, params, sim_tuple, sub_traj, forward, stop_grad)
    return -1.0 * w.mean(), (w, samples_new)


def sub_traj_fwd_tb(
    keys,
    samples,
    samples_next,
    model_state,
    params,
    sim_tuple,
    traj_start,
    traj_end,
    traj_idx,
    traj_length,
    forward=False,
    stop_grad=True,
    per_sample_rnd_fn=per_sample_sub_traj_is_weight,
):
    sub_traj = traj_start, traj_end, traj_idx, traj_length
    log_w, samples_new = jax.vmap(
        per_sample_rnd_fn, in_axes=(0, 0, None, None, None, None, None, None)
    )(keys, samples_next, model_state, params, sim_tuple, sub_traj, forward, stop_grad)
    tb_vals = jnp.mean(jnp.square(log_w - params["params"]["logZ_deltas"][traj_idx]))
    return tb_vals, (log_w, samples_new)


def sub_traj_rev_tb(
    keys,
    samples,
    samples_next,
    model_state,
    params,
    sim_tuple,
    traj_start,
    traj_end,
    traj_idx,
    traj_length,
    forward=True,
    stop_grad=True,
    f_fn=jnp.square,
    per_sample_rnd_fn=per_sample_sub_traj_is_weight,
):
    sub_traj = traj_start, traj_end, traj_idx, traj_length

    log_w, samples_new = jax.vmap(
        per_sample_rnd_fn, in_axes=(0, 0, None, None, None, None, None, None)
    )(keys, samples, model_state, params, sim_tuple, sub_traj, forward, stop_grad)

    lnZ_delta = jax.lax.stop_gradient(params["params"]["logZ_deltas"][traj_idx])
    tb_vals = jnp.mean(f_fn(log_w - lnZ_delta))
    return tb_vals, (log_w, samples_new)


def sub_traj_fwd_logvar(
    keys,
    samples,
    samples_next,
    model_state,
    params,
    sim_tuple,
    traj_start,
    traj_end,
    traj_idx,
    traj_length,
    forward=False,
    stop_grad=True,
    per_sample_rnd_fn=per_sample_sub_traj_is_weight,
):

    sub_traj = traj_start, traj_end, traj_idx, traj_length
    w, samples_new = jax.vmap(
        per_sample_rnd_fn, in_axes=(0, 0, None, None, None, None, None, None)
    )(keys, samples_next, model_state, params, sim_tuple, sub_traj, forward, stop_grad)
    return jnp.clip(w.var(ddof=0), -1e7, 1e7), (w, samples_new)


def sub_traj_rev_logvar(
    keys,
    samples,
    samples_next,
    model_state,
    params,
    sim_tuple,
    traj_start,
    traj_end,
    traj_idx,
    traj_length,
    forward=True,
    stop_grad=True,
    detach_langevin_pisgrad=True,
    per_sample_rnd_fn=per_sample_sub_traj_is_weight,
):
    sub_traj = traj_start, traj_end, traj_idx, traj_length

    w, samples_new = jax.vmap(
        per_sample_rnd_fn, in_axes=(0, 0, None, None, None, None, None, None, None)
    )(
        keys,
        samples,
        model_state,
        params,
        sim_tuple,
        sub_traj,
        forward,
        stop_grad,
        detach_langevin_pisgrad,
    )
    return jnp.clip(w.var(ddof=0), -1e7, 1e7), (w, samples_new)


def get_loss_fn(identifier: str):
    if identifier == "fwd_kl":
        return sub_traj_fwd_kl
    elif identifier == "rev_kl":
        return sub_traj_rev_kl
    elif identifier == "fwd_tb":
        return sub_traj_fwd_tb
    elif identifier == "rev_tb":
        return sub_traj_rev_tb
    elif identifier == "fwd_lv":
        return sub_traj_fwd_logvar
    elif identifier == "rev_lv":
        return sub_traj_rev_logvar
    else:
        raise ValueError(f"{identifier} is not a valid identifier for a loss function")
