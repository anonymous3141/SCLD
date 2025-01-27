from functools import partial

import jax
import jax.numpy as jnp
from algorithms.scld import resampling
from algorithms.scld.scld_utils import log_prob_kernel, sample_kernel
from jax.experimental.ode import odeint
from jax.scipy.special import logsumexp


def clip_langevin(langevin, clip_thresh):
    langevin_norm = jnp.linalg.norm(langevin)
    clipped_norm = jnp.clip(langevin_norm, min=-clip_thresh, max=clip_thresh)
    norm_ratio = jax.lax.stop_gradient(clipped_norm / langevin_norm)
    return langevin * norm_ratio


def per_subtraj_log_is(
    key,
    traj,
    model_state,
    params,
    sim_tuple,
    sub_traj,
    forward=True,
    stop_grad=False,
    detach_langevin_pisgrad=True,
):
    """
    Input: trajectory or subtrajectory and start & endpoints of subtrajectory
    Output: the logrnds of the sub-trajectory according to current drift network

    (we require the same signature as rnd fn for samples for consistency)
    Note: traj is the actual trajectory, sub_traj has details about where the trajectory
    starts and ends
    """

    (log_density_per_step, noise_schedule, total_steps, (langevin_norm_clip,)) = (
        sim_tuple
    )
    dt = 1.0 / total_steps

    def aggregate_rnd(current_state, xs):
        (running_rnd, step) = current_state
        (x, x_new) = xs
        # Compute forward & backward means
        sigma_t = noise_schedule(step)
        scale = sigma_t * jnp.sqrt(2 * dt)
        langevin = clip_langevin(
            jax.grad(log_density_per_step, 1)(step, x), langevin_norm_clip
        )
        langevin_detached = jax.lax.stop_gradient(langevin)
        model_output = model_state.apply_fn(
            params,
            x,
            step * jnp.ones(1),
            langevin_detached if detach_langevin_pisgrad else langevin,
        )
        # Euler-Maruyama integration of the SDE
        fwd_mean = x + sigma_t**2 * (langevin + model_output) * dt

        # update backward mean
        langevin_new = clip_langevin(
            jax.grad(log_density_per_step, 1)(step, x_new), langevin_norm_clip
        )
        langevin_new_detached = jax.lax.stop_gradient(langevin_new)
        model_output_new = model_state.apply_fn(
            params,
            x_new,
            (step + 1) * jnp.ones(1),
            langevin_new_detached if detach_langevin_pisgrad else langevin_new,
        )

        bwd_mean = x_new + sigma_t**2 * (langevin_new - model_output_new) * dt

        fwd_log_prob = log_prob_kernel(
            x_new, fwd_mean, scale
        )  # log prob of the forward step
        bwd_log_prob = log_prob_kernel(
            x, bwd_mean, scale
        )  # log prob of the backward step

        nxt_rnd = running_rnd + bwd_log_prob - fwd_log_prob
        return (nxt_rnd, step + 1), nxt_rnd

    traj_start, traj_end, traj_idx, traj_length = sub_traj
    initial_state = (0, traj_start.squeeze())
    # Stack the two arrays along a new dimension
    traj_pairs = jnp.stack((traj[:-1], traj[1:]), axis=1)

    (log_rnd, final_time), _ = jax.lax.scan(aggregate_rnd, initial_state, traj_pairs)

    final_log_prob = log_density_per_step(traj_end, traj[-1])
    init_log_prob = log_density_per_step(traj_start, traj[0])
    log_rnd += final_log_prob - init_log_prob
    return log_rnd, None


def per_sample_sub_traj_is_weight(
    key,
    x_start,
    model_state,
    params,
    sim_tuple,
    sub_traj,
    forward=True,
    stop_grad=False,
    detach_langevin_pisgrad=True,
):
    """
    Computes the incremental importance sampling weights for a single sample x.

    Input:
     - subtrajectory section to simulate
     - start point
     - simulation parameters (annealing schedule, noise schedule etc)

    Output:
    - final point, rnd
    """

    # gets details of noise schedule, annealing schedule = log_density_per_step, etc
    (log_density_per_step, noise_schedule, total_steps, (langevin_norm_clip,)) = (
        sim_tuple
    )

    dt = 1.0 / total_steps

    def is_weight_forward(state, step):
        """
        Takes samples from π_{t} and moves them to π_{t+1}. Computes the incremental IS weight.
        """
        x, log_w, key_gen = state

        step = step.astype(jnp.float32)

        # Compute SDE components
        sigma_t = noise_schedule(step)
        scale = sigma_t * jnp.sqrt(2 * dt)
        langevin = clip_langevin(
            jax.grad(log_density_per_step, 1)(step, x), langevin_norm_clip
        )
        langevin_detached = jax.lax.stop_gradient(langevin)
        model_output = model_state.apply_fn(
            params,
            x,
            step * jnp.ones(1),
            langevin_detached if detach_langevin_pisgrad else langevin,
        )

        # Euler-Maruyama integration of the SDE
        fwd_mean = x + sigma_t**2 * (langevin + model_output) * dt

        key, key_gen = jax.random.split(key_gen)
        x_new = sample_kernel(
            key, jax.lax.stop_gradient(fwd_mean) if stop_grad else fwd_mean, scale
        )

        langevin_new = clip_langevin(
            jax.grad(log_density_per_step, 1)(step, x_new), langevin_norm_clip
        )
        langevin_new_detached = jax.lax.stop_gradient(langevin_new)
        model_output_new = model_state.apply_fn(
            params,
            x_new,
            (step + 1) * jnp.ones(1),
            langevin_new_detached if detach_langevin_pisgrad else langevin_new,
        )

        bwd_mean = x_new + sigma_t**2 * (langevin_new - model_output_new) * dt

        fwd_log_prob = log_prob_kernel(
            x_new, fwd_mean, scale
        )  # log prob of the forward step
        bwd_log_prob = log_prob_kernel(
            x, bwd_mean, scale
        )  # log prob of the backward step

        key, key_gen = jax.random.split(key_gen)
        log_w += bwd_log_prob - fwd_log_prob
        next_state = (x_new, log_w, key_gen)
        per_step_output = (x_new,)
        return next_state, per_step_output

    def is_weight_backward(state, step):
        """
        Takes samples from π_{t+1} and moves them to π_{t}. Computes the incremental IS weight.
        """
        x, log_w, key_gen = state
        next_step = step + 1

        step = step.astype(jnp.float32)
        if stop_grad:
            x = jax.lax.stop_gradient(x)

        # Compute SDE components
        sigma_t = noise_schedule(next_step)
        scale = sigma_t * jnp.sqrt(2 * dt)
        langevin = jax.grad(log_density_per_step, 1)(next_step, x)
        langevin_detached = jax.lax.stop_gradient(langevin)
        model_output = model_state.apply_fn(
            params,
            x,
            next_step * jnp.ones(1),
            langevin_detached if detach_langevin_pisgrad else langevin,
        )

        # Euler-Maruyama integration of the SDE
        # note: We detach the score when feeding to model, but not when feeding to mean
        bwd_mean = x + sigma_t**2 * (langevin - model_output) * dt

        key, key_gen = jax.random.split(key_gen)
        x_new = sample_kernel(key, bwd_mean, scale)

        if stop_grad:
            x_new = jax.lax.stop_gradient(x_new)

        langevin_new = jax.grad(log_density_per_step, 1)(next_step, x_new)
        langevin_new_detached = jax.lax.stop_gradient(langevin_new)
        model_output_new = model_state.apply_fn(
            params,
            x_new,
            step * jnp.ones(1),
            langevin_new_detached if detach_langevin_pisgrad else langevin_new,
        )
        fwd_mean = x_new + sigma_t**2 * (langevin_new + model_output_new) * dt

        fwd_log_prob = log_prob_kernel(x, fwd_mean, scale)
        bwd_log_prob = log_prob_kernel(x_new, bwd_mean, scale)

        key, key_gen = jax.random.split(key_gen)
        log_w += bwd_log_prob - fwd_log_prob
        next_state = (x_new, log_w, key_gen)
        per_step_output = (x_new,)
        return next_state, per_step_output

    traj_start, traj_end, traj_idx, traj_length = sub_traj

    rng_key, rng_key_gen = jax.random.split(key)
    initial_state = (x_start, 0, rng_key_gen)

    trajectories = None
    if forward:
        final_state, trajectories = jax.lax.scan(
            is_weight_forward, initial_state, traj_start + jnp.arange(traj_length)
        )
        x_final, delta_w, _ = final_state
        final_log_prob, init_log_prob = log_density_per_step(
            traj_end, x_final
        ), log_density_per_step(traj_start, x_start)
    else:
        final_state, trajectories = jax.lax.scan(
            is_weight_backward, initial_state, jnp.arange(traj_length)[::-1]
        )
        x_final, delta_w, _ = final_state
        final_log_prob, init_log_prob = log_density_per_step(
            traj_end, x_start
        ), log_density_per_step(traj_start, x_final)
    # note: trajectories detaches gradient
    trajectories = jax.lax.stop_gradient(
        jnp.concatenate([jnp.expand_dims(x_start, 0), trajectories[0]], axis=0)
    )
    delta_w += final_log_prob - init_log_prob

    return delta_w, (x_final, final_log_prob, trajectories)


# vectorized version of above
def sub_traj_is_weights(
    keys,
    samples,
    model_state,
    params,
    sim_tuple,
    sub_traj,
    forward=True,
    stop_grad=True,
    detach_langevin_pisgrad=True,
):
    """
    Computes the incremental importance weights of a sub-trajectory, i.e.,
    G(x_t, x_t+1) = γ(x_{t+1}) B(x_{t}|x_{t+1}) / γ(x_{t}) F(x_{t+1}|x_{t})
    """
    w, aux = jax.vmap(
        per_sample_sub_traj_is_weight,
        in_axes=(0, 0, None, None, None, None, None, None, None),
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
    return (
        w.reshape(
            -1,
        ),
        aux,
    )


def get_lnz_elbo_increment(log_is_weights, log_weights):
    """
    compute updates to lnZ and ELBO
    """
    normalized_log_weights = jax.nn.log_softmax(log_weights)
    total_terms = jax.lax.stop_gradient(normalized_log_weights) + log_is_weights
    lnz_inc = logsumexp(total_terms)

    elbo_inc = jnp.sum(
        jax.lax.stop_gradient(jnp.exp(normalized_log_weights)) * log_is_weights
    )
    return lnz_inc, elbo_inc


def update_samples_log_weights(
    samples,
    log_is_weights,
    log_weights,
    markov_kernel_apply,
    key,
    step: int,
    use_reweighting: bool,
    use_resampling: bool,
    resampler,
    use_markov: bool,
    resample_threshold: float,
    log_density_per_step,
):
    """
    Applies Resampling and MCMC steps
    Computes new Particle weights
    """

    if use_reweighting:
        log_weights_new = reweight(log_weights, log_is_weights)
    else:
        log_weights_new = log_weights

    # do resampling if applicable
    if use_resampling:
        subkey, key = jax.random.split(key)
        resampled_samples, log_weights_resampled = resampling.optionally_resample(
            subkey, log_weights_new, samples, resample_threshold, resampler
        )
    else:
        resampled_samples = samples
        log_weights_resampled = log_weights_new

    # Apply the Markov correction steps that maps resampled_samples to markov_samples
    if use_markov:
        markov_samples, acceptance_tuple = markov_kernel_apply(
            step, key, resampled_samples
        )
    else:
        markov_samples = resampled_samples
        acceptance_tuple = (1.0, 1.0, 1.0)

    return markov_samples, log_weights_resampled, acceptance_tuple, (log_weights_new,)


def reweight(log_weights_old, log_is_weights):
    log_weights_new_unorm = log_weights_old + log_is_weights
    log_weights_new = jax.nn.log_softmax(log_weights_new_unorm)
    return log_weights_new
