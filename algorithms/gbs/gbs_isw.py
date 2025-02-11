from functools import partial

import jax
import jax.numpy as jnp
import numpyro.distributions as npdist


def sample_kernel(rng_key, mean, scale):
    eps = jax.random.normal(rng_key, shape=(mean.shape[0],))
    return mean + scale * eps


def log_prob_kernel(x, mean, scale):
    dist = npdist.Independent(npdist.Normal(loc=mean, scale=scale), 1)
    return dist.log_prob(x)


def per_sample_rnd(
    seed,
    model_state,
    fwd_params,
    bwd_params,
    aux_tuple,
    target,
    num_steps,
    noise_schedule,
    stop_grad=False,
    prior_to_target=True,
):
    prior_sampler, prior_log_prob = aux_tuple
    target_log_prob = target.log_prob

    fwd_state, bwd_state = model_state

    def langevin_score_fn(x, step, initial_log_prob, target_log_prob):
        beta = step / num_steps
        return beta * target_log_prob(x) + (1 - beta) * initial_log_prob(x)

    sigmas = noise_schedule

    langevin_score = partial(
        langevin_score_fn,
        initial_log_prob=prior_log_prob,
        target_log_prob=target_log_prob,
    )
    dt = 1.0 / num_steps

    def simulate_prior_to_target(state, per_step_input):
        """
        Takes samples from the prior and moves them to the target
        """

        x, log_w, key_gen = state
        step = per_step_input

        step = step.astype(jnp.float32)
        if stop_grad:
            x = jax.lax.stop_gradient(x)

        # Compute SDE components
        sigma_t = sigmas(step)
        scale = sigma_t * jnp.sqrt(2 * dt)
        fwd_langevin = jax.grad(langevin_score)(x, step)
        fwd_langevin_detached = jax.lax.stop_gradient(fwd_langevin)
        fwd_model = fwd_state.apply_fn(
            fwd_params, x, step * jnp.ones(1), fwd_langevin_detached
        )

        # Euler-Maruyama integration of the SDE
        fwd_mean = x + sigma_t**2 * fwd_model * dt  # todo check the coefficients

        key, key_gen = jax.random.split(key_gen)
        x_new = sample_kernel(key, fwd_mean, scale)

        if stop_grad:
            x_new = jax.lax.stop_gradient(x_new)

        bwd_langevin_new = jax.grad(langevin_score)(x_new, step + 1)
        bwd_langevin_new_detached = jax.lax.stop_gradient(bwd_langevin_new)
        bwd_model = bwd_state.apply_fn(
            bwd_params, x_new, (step + 1) * jnp.ones(1), bwd_langevin_new_detached
        )

        bwd_mean = x_new + sigma_t**2 * bwd_model * dt

        fwd_log_prob = log_prob_kernel(x_new, fwd_mean, scale)
        bwd_log_prob = log_prob_kernel(x, bwd_mean, scale)

        key, key_gen = jax.random.split(key_gen)
        log_w += bwd_log_prob - fwd_log_prob
        next_state = (x_new, log_w, key_gen)
        per_step_output = (x_new,)
        return next_state, per_step_output

    def simulate_target_to_prior(state, per_step_input):
        """
        Takes samples from the target and moves them to the prior
        """

        x, log_w, key_gen = state
        step = per_step_input
        next_step = step + 1

        step = step.astype(jnp.float32)
        if stop_grad:
            x = jax.lax.stop_gradient(x)

        # Compute SDE components
        sigma_t = sigmas(next_step)
        scale = sigma_t * jnp.sqrt(2 * dt)
        bwd_langevin_new = jax.grad(langevin_score)(x, step + 1)
        bwd_langevin_new_detached = jax.lax.stop_gradient(bwd_langevin_new)
        bwd_model = bwd_state.apply_fn(
            bwd_params, x, (step + 1) * jnp.ones(1), bwd_langevin_new_detached
        )
        key, key_gen = jax.random.split(key_gen)

        # Euler-Maruyama integration of the SDE
        bwd_mean = x + sigma_t**2 * bwd_model * dt

        key, key_gen = jax.random.split(key_gen)
        x_new = sample_kernel(key, bwd_mean, scale)

        if stop_grad:
            x_new = jax.lax.stop_gradient(x_new)

        fwd_langevin = jax.grad(langevin_score)(x_new, step)
        fwd_langevin_detached = jax.lax.stop_gradient(fwd_langevin)
        fwd_model = fwd_state.apply_fn(
            fwd_params, x_new, step * jnp.ones(1), fwd_langevin_detached
        )
        fwd_mean = x_new + sigma_t**2 * fwd_model * dt

        fwd_log_prob = log_prob_kernel(x, fwd_mean, scale)
        bwd_log_prob = log_prob_kernel(x_new, bwd_mean, scale)

        key, key_gen = jax.random.split(key_gen)
        log_w += bwd_log_prob - fwd_log_prob
        next_state = (x_new, log_w, key_gen)
        per_step_output = (x_new,)
        return next_state, per_step_output

    key, key_gen = jax.random.split(seed)
    key, key_gen = jax.random.split(key_gen)
    if prior_to_target:
        init_x = jnp.squeeze(prior_sampler(seed=key, sample_shape=(1,)))
        aux = (init_x, 0.0, key)
        aux, per_step_output = jax.lax.scan(
            simulate_prior_to_target, aux, jnp.arange(0, num_steps)
        )
        final_x, log_ratio, _ = aux
        terminal_cost = prior_log_prob(init_x) - target_log_prob(final_x)
    else:
        init_x = jnp.squeeze(target.sample(key, (1,)))
        aux = (init_x, 0.0, key)
        aux, per_step_output = jax.lax.scan(
            simulate_target_to_prior, aux, jnp.arange(0, num_steps)[::-1]
        )
        final_x, log_ratio, _ = aux
        terminal_cost = prior_log_prob(final_x) - target_log_prob(init_x)

    running_cost = -log_ratio
    x_t = per_step_output
    stochastic_costs = jnp.zeros_like(running_cost)
    return final_x, running_cost, stochastic_costs, terminal_cost, x_t


def rnd(
    key,
    model_state,
    fwd_params,
    bwd_params,
    batch_size,
    aux_tuple,
    target,
    num_steps,
    noise_schedule,
    stop_grad=False,
    prior_to_target=True,
):
    seeds = jax.random.split(key, num=batch_size)
    x_0, running_costs, stochastic_costs, terminal_costs, x_t = jax.vmap(
        per_sample_rnd,
        in_axes=(0, None, None, None, None, None, None, None, None, None),
    )(
        seeds,
        model_state,
        fwd_params,
        bwd_params,
        aux_tuple,
        target,
        num_steps,
        noise_schedule,
        stop_grad,
        prior_to_target,
    )

    return x_0, running_costs, stochastic_costs, terminal_costs


def neg_elbo(
    key,
    model_state,
    fwd_params,
    bwd_params,
    batch_size,
    aux_tuple,
    target,
    num_steps,
    noise_schedule,
    stop_grad=False,
):
    aux = rnd(
        key,
        model_state,
        fwd_params,
        bwd_params,
        batch_size,
        aux_tuple,
        target,
        num_steps,
        noise_schedule,
        stop_grad,
    )
    samples, running_costs, stochastic_costs, terminal_costs = aux
    neg_elbo = running_costs + terminal_costs
    return jnp.mean(neg_elbo), (neg_elbo, samples)
