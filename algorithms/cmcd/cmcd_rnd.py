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
    params,
    aux_tuple,
    target,
    num_steps,
    noise_schedule,
    stop_grad=True,
    prior_to_target=True,
    paths=None,
):
    prior_sampler, prior_log_prob, get_betas = aux_tuple
    target_log_prob = target.log_prob

    def langevin_score_fn(x, beta, params, initial_log_prob, target_log_prob):
        return beta * target_log_prob(x) + (1 - beta) * initial_log_prob(params, x)

    sigmas = noise_schedule
    betas = get_betas(params)

    langevin_score = partial(
        langevin_score_fn,
        initial_log_prob=prior_log_prob,
        target_log_prob=target_log_prob,
    )

    def stable_langevin(x, beta, params):
        p = lambda x: prior_log_prob(params, x)
        grad_p = jax.grad(p)(x)
        u = lambda x: target_log_prob(x)
        grad_u = jax.grad(u)(x)
        grad_u_clipped = jnp.clip(grad_u, -1e3, 1e3)
        return beta * grad_u_clipped + (1 - beta) * grad_p

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
        beta_t = betas(step)
        scale = sigma_t * jnp.sqrt(2 * dt)
        # langevin = jax.grad(langevin_score)(x, beta_t, params)
        langevin = stable_langevin(x, beta_t, params)
        langevin_detached = jax.lax.stop_gradient(langevin)
        model_output = model_state.apply_fn(
            params, x, step * jnp.ones(1), langevin_detached
        )

        # Euler-Maruyama integration of the SDE
        fwd_mean = x + sigma_t**2 * (langevin + model_output) * dt

        key, key_gen = jax.random.split(key_gen)
        x_new = sample_kernel(key, fwd_mean, scale)

        if stop_grad:
            x_new = jax.lax.stop_gradient(x_new)

        # langevin_new = jax.grad(langevin_score)(x_new, beta_t, params)
        langevin_new = stable_langevin(x_new, beta_t, params)
        langevin_new_detached = jax.lax.stop_gradient(langevin_new)
        model_output_new = model_state.apply_fn(
            params, x_new, (step + 1) * jnp.ones(1), langevin_new_detached
        )

        bwd_mean = x_new + sigma_t**2 * (langevin_new - model_output_new) * dt

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
        beta_t = betas(next_step)
        scale = sigma_t * jnp.sqrt(2 * dt)
        # langevin = jax.grad(langevin_score)(x, beta_t, params)
        langevin = stable_langevin(x, beta_t, params)
        langevin_detached = jax.lax.stop_gradient(langevin)
        model_output = model_state.apply_fn(
            params, x, next_step * jnp.ones(1), langevin_detached
        )
        key, key_gen = jax.random.split(key_gen)

        # Euler-Maruyama integration of the SDE
        bwd_mean = x + sigma_t**2 * (langevin - model_output) * dt

        key, key_gen = jax.random.split(key_gen)
        x_new = sample_kernel(key, bwd_mean, scale)

        if stop_grad:
            x_new = jax.lax.stop_gradient(x_new)

        # langevin_new = jax.grad(langevin_score)(x_new, beta_t, params)
        langevin_new = stable_langevin(x_new, beta_t, params)
        langevin_new_detached = jax.lax.stop_gradient(langevin_new)
        model_output_new = model_state.apply_fn(
            params, x_new, step * jnp.ones(1), langevin_new_detached
        )
        fwd_mean = x_new + sigma_t**2 * (langevin_new + model_output_new) * dt

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
        init_x = jnp.squeeze(prior_sampler(params, key, 1))
        aux = (init_x, 0.0, key)
        aux, per_step_output = jax.lax.scan(
            simulate_prior_to_target, aux, jnp.arange(0, num_steps)
        )

        final_x, log_ratio, _ = aux
        terminal_cost = prior_log_prob(params, init_x) - target_log_prob(final_x)
    else:
        init_x = jnp.squeeze(target.sample(key, (1,)))
        aux = (init_x, 0.0, key)
        aux, per_step_output = jax.lax.scan(
            simulate_target_to_prior, aux, jnp.arange(0, num_steps)[::-1]
        )
        final_x, log_ratio, _ = aux
        terminal_cost = prior_log_prob(params, final_x) - target_log_prob(init_x)

    running_cost = -log_ratio
    x_t = per_step_output
    stochastic_costs = jnp.zeros_like(running_cost)
    return final_x, running_cost, stochastic_costs, terminal_cost, x_t


def per_sample_eval(
    paths,
    model_state,
    params,
    aux_tuple,
    target,
    num_steps,
    noise_schedule,
    prior_to_target=True,
):
    _, prior_log_prob, get_betas = aux_tuple
    target_log_prob = target.log_prob

    sigmas = noise_schedule
    betas = get_betas(params)

    def stable_langevin(x, beta, params):
        p = lambda x: prior_log_prob(params, x)
        grad_p = jax.grad(p)(x)
        u = lambda x: target_log_prob(x)
        grad_u = jax.grad(u)(x)
        grad_u_clipped = jnp.clip(grad_u, -1e3, 1e3)
        return beta * grad_u_clipped + (1 - beta) * grad_p

    dt = 1.0 / num_steps

    def eval_prior_to_target(x, x_new):
        """
        Takes samples from the prior and moves them to the target
        """
        # x = paths[step]
        # x_new = paths[step + 1]
        step = jax.lax.axis_index("step")
        x = jax.lax.stop_gradient(x)
        x_new = jax.lax.stop_gradient(x_new)

        sigma_t = sigmas(step)
        beta_t = betas(step)

        scale = sigma_t * jnp.sqrt(2 * dt)
        # langevin = jax.grad(langevin_score)(x, beta_t, params)
        langevin = stable_langevin(x, beta_t, params)
        langevin_detached = jax.lax.stop_gradient(langevin)
        model_output = model_state.apply_fn(
            params, x, step * jnp.ones(1), langevin_detached
        )

        # Euler-Maruyama integration of the SDE
        fwd_mean = x + sigma_t**2 * (langevin + model_output) * dt

        # langevin_new = jax.grad(langevin_score)(x_new, beta_t, params)
        langevin_new = stable_langevin(x_new, beta_t, params)
        langevin_new_detached = jax.lax.stop_gradient(langevin_new)
        model_output_new = model_state.apply_fn(
            params, x_new, (step + 1) * jnp.ones(1), langevin_new_detached
        )

        bwd_mean = x_new + sigma_t**2 * (langevin_new - model_output_new) * dt

        fwd_log_prob = log_prob_kernel(x_new, fwd_mean, scale)
        bwd_log_prob = log_prob_kernel(x, bwd_mean, scale)

        log_w_t = bwd_log_prob - fwd_log_prob

        return log_w_t

    if prior_to_target:
        log_ratios = jax.vmap(eval_prior_to_target, axis_name="step")(
            paths[:-1], paths[1:]
        )

        running_cost = -jnp.sum(log_ratios)
        terminal_cost = prior_log_prob(params, paths[0]) - target_log_prob(paths[-1])
    else:
        raise ValueError("Not implemented")

    stochastic_costs = jnp.zeros_like(running_cost)
    return running_cost, stochastic_costs, terminal_cost


def rnd(
    key,
    model_state,
    params,
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
        per_sample_rnd, in_axes=(0, None, None, None, None, None, None, None, None)
    )(
        seeds,
        model_state,
        params,
        aux_tuple,
        target,
        num_steps,
        noise_schedule,
        stop_grad,
        prior_to_target,
    )

    return x_0, running_costs, stochastic_costs, terminal_costs, x_t


def eval(
    paths,
    model_state,
    params,
    aux_tuple,
    target,
    num_steps,
    noise_schedule,
    prior_to_target=True,
):
    running_costs, stochastic_costs, terminal_costs = jax.vmap(
        per_sample_eval, in_axes=(0, None, None, None, None, None, None, None)
    )(
        paths,
        model_state,
        params,
        aux_tuple,
        target,
        num_steps,
        noise_schedule,
        prior_to_target,
    )

    return running_costs + terminal_costs


def neg_elbo(
    key,
    model_state,
    params,
    batch_size,
    aux_tuple,
    target,
    num_steps,
    noise_schedule,
    buffer,
    ln_z,
):
    del buffer, ln_z
    aux = rnd(
        key,
        model_state,
        params,
        batch_size,
        aux_tuple,
        target,
        num_steps,
        noise_schedule,
        False,
    )
    samples, running_costs, stochastic_costs, terminal_costs, _ = aux
    neg_elbo = running_costs + terminal_costs
    return jnp.mean(neg_elbo), (neg_elbo, samples, None, None, 0.0)


def log_var(
    key,
    model_state,
    params,
    batch_size,
    aux_tuple,
    target,
    num_steps,
    noise_schedule,
    buffer,
    ln_z,
):
    del ln_z
    aux = rnd(
        key,
        model_state,
        params,
        batch_size,
        aux_tuple,
        target,
        num_steps,
        noise_schedule,
        True,
    )
    samples, running_costs, stochastic_costs, terminal_costs, trajectories = aux

    trajectories = trajectories[0]

    # Small ELBO is high quality trajectory. Therefore, large neg_elbo is high quality trajectory.
    neg_elbo = running_costs + terminal_costs

    if buffer is None:
        return jnp.clip((neg_elbo).var(ddof=0), -1e7, 1e7), (
            neg_elbo,
            samples,
            trajectories,
            None,
            0.0,
        )
    else:
        # pick batch_size // 2 points from buffer proportional to their neg_elbo
        sampling_key = jax.random.split(key, num=2)[0]

        # Sample trajectories from buffer, higher neg_elbo is better.
        eval_trajectories, _, eval_idx = buffer.sample(sampling_key, batch_size)

        neg_elbo_eval = eval(
            eval_trajectories,
            model_state,
            params,
            aux_tuple,
            target,
            num_steps,
            noise_schedule,
            True,
        )

        buffer.update_elbos(eval_idx, neg_elbo_eval)
        # After sampling from buffer, add the new trajectories to the buffer
        buffer.update(trajectories, neg_elbo)

        neg_elbo = jnp.concatenate([neg_elbo, neg_elbo_eval], axis=0)

        return jnp.clip((neg_elbo).var(ddof=0), -1e7, 1e7), (
            neg_elbo,
            samples,
            trajectories,
            buffer,
            0.0,
        )


def traj_bal(
    key,
    model_state,
    params,
    batch_size,
    aux_tuple,
    target,
    num_steps,
    noise_schedule,
    buffer,
    ln_z,
):
    del ln_z
    aux = rnd(
        key,
        model_state,
        params,
        batch_size,
        aux_tuple,
        target,
        num_steps,
        noise_schedule,
        True,
    )
    samples, running_costs, stochastic_costs, terminal_costs, trajectories = aux

    trajectories = trajectories[0]

    # Small ELBO is high quality trajectory. Therefore, large neg_elbo is high quality trajectory.
    neg_elbo = running_costs + terminal_costs

    if buffer is None:
        return jnp.mean((neg_elbo + params["params"]["traj_bal"]) ** 2), (
            neg_elbo,
            samples,
            trajectories,
            None,
            0.0,
        )
    else:
        # pick batch_size // 2 points from buffer proportional to their neg_elbo
        sampling_key = jax.random.split(key, num=2)[0]

        # Sample trajectories from buffer, higher neg_elbo is better.
        eval_trajectories, _, eval_idx = buffer.sample(sampling_key, batch_size)

        neg_elbo_eval = eval(
            eval_trajectories,
            model_state,
            params,
            aux_tuple,
            target,
            num_steps,
            noise_schedule,
            True,
        )

        buffer.update_elbos(eval_idx, neg_elbo_eval)
        # After sampling from buffer, add the new trajectories to the buffer
        buffer.update(trajectories, neg_elbo)

        neg_elbo = jnp.concatenate([neg_elbo, neg_elbo_eval], axis=0)

        return jnp.mean((neg_elbo + params["params"]["traj_bal"]) ** 2), (
            neg_elbo,
            samples,
            trajectories,
            buffer,
            0.0,
        )


def traj_bal_jensens(
    key,
    model_state,
    params,
    batch_size,
    aux_tuple,
    target,
    num_steps,
    noise_schedule,
    buffer,
    ln_z,
    f_fn=lambda x: x * x,
):  # Brian's f_divergence framework
    aux = rnd(
        key,
        model_state,
        params,
        batch_size,
        aux_tuple,
        target,
        num_steps,
        noise_schedule,
        True,
    )
    samples, running_costs, _, terminal_costs, trajectories = aux

    trajectories = trajectories[0]
    neg_elbo = running_costs + terminal_costs

    if buffer is None:
        alpha = 0.1

        # TODO: Init at mean(exp(elbo))
        ln_z = (1 - alpha) * ln_z + alpha * jax.scipy.special.logsumexp(
            -neg_elbo - jnp.log(batch_size)
        )
        ln_z = jax.lax.stop_gradient(ln_z)

        return jnp.mean(f_fn(neg_elbo + ln_z)), (
            neg_elbo,
            samples,
            trajectories,
            None,
            ln_z,
        )
    else:
        # pick batch_size // 2 points from buffer proportional to their neg_elbo
        sampling_key = jax.random.split(key, num=2)[0]

        # Sample trajectories from buffer, higher neg_elbo is better.
        eval_trajectories, _, eval_idx = buffer.sample(sampling_key, batch_size)

        neg_elbo_eval = eval(
            eval_trajectories,
            model_state,
            params,
            aux_tuple,
            target,
            num_steps,
            noise_schedule,
            True,
        )

        buffer.update_elbos(eval_idx, neg_elbo_eval)
        # After sampling from buffer, add the new trajectories to the buffer
        buffer.update(trajectories, neg_elbo)

        alpha = 0.1

        # TODO: Init at mean(exp(elbo))
        ln_z = (1 - alpha) * ln_z + alpha * jax.scipy.special.logsumexp(
            -neg_elbo - jnp.log(batch_size)
        )
        ln_z = jax.lax.stop_gradient(ln_z)

        neg_elbo = jnp.concatenate([neg_elbo, neg_elbo_eval], axis=0)

        return jnp.mean(f_fn(neg_elbo + ln_z)), (
            neg_elbo,
            samples,
            trajectories,
            buffer,
            ln_z,
        )
