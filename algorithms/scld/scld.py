import time
from functools import partial

import algorithms.common.types as tp
import chex
import distrax
import jax
import jax.numpy as jnp
import optax
import wandb
from algorithms.common import markov_kernel
from algorithms.common.ipm_eval import discrepancies
from algorithms.common.models.pisgrad_net import PISGRADNet
from algorithms.common.models.statetime_net import StateTimeNetwork
from algorithms.scld import mfvi, resampling
from algorithms.scld.is_weights import (get_lnz_elbo_increment,
                                        per_subtraj_log_is,
                                        sub_traj_is_weights,
                                        update_samples_log_weights)
from algorithms.scld.loss_fns import (get_loss_fn, lnZ_update_jensen,
                                      lnZ_update_vanilla, sub_traj_rev_kl)
from algorithms.scld.prioritised_buffer import build_prioritised_subtraj_buffer
from algorithms.scld.scld_eval import eval_scld
from algorithms.scld.scld_utils import (GeometricAnnealingSchedule,
                                        flattened_traversal, gradient_step,
                                        make_lr_scheduler, print_results)
from flax.training import train_state

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


def f_cosine(x):
    s = 0.008
    return jnp.cos((x + s) / (1 + s) * jnp.pi / 2) ** 2


def get_beta_nonlearnt(step, alg_cfg):
    if alg_cfg.annealing_schedule.schedule_type == "uniform":
        beta = step / alg_cfg.num_steps
        return beta
    elif alg_cfg.annealing_schedule.schedule_type == "cosine":
        # eq17 of https://arxiv.org/pdf/2102.09672
        return f_cosine(1 - step / alg_cfg.num_steps) / f_cosine(0)
    else:
        raise NotImplementedError


def get_beta_schedule(params, alg_cfg):
    if alg_cfg.annealing_schedule.schedule_type == "learnt":
        # for dealing with learnt betas
        b = jax.nn.softplus(params["params"]["betas"])
        b = jnp.cumsum(b) / jnp.sum(b)
        b = jnp.concatenate((jnp.array([0]), b))

        def get_beta(step):
            return b[jnp.array(step, int)]

        return get_beta
    else:
        return partial(get_beta_nonlearnt, alg_cfg=alg_cfg)


def get_annealing_fn_and_prior(params, alg_cfg, target_log_prob):
    initial_density = distrax.MultivariateNormalDiag(
        params["params"]["prior_mean"], jnp.exp(params["params"]["prior_log_stds"])
    )

    fixed_annealing_schedule = GeometricAnnealingSchedule(
        initial_density.log_prob,
        target_log_prob,
        alg_cfg.num_steps + 1,
        alg_cfg.target_clip,
        alg_cfg.annealing_schedule.schedule_type,
    )

    beta_fn = get_beta_schedule(params, alg_cfg)

    def get_schedule(step, x):
        log_densities_final = fixed_annealing_schedule._final_log_density(x)
        log_densities_initial = fixed_annealing_schedule._initial_log_density(x)
        beta = beta_fn(step)
        return (1.0 - beta) * log_densities_initial + beta * log_densities_final

    noise_schedule = (
        alg_cfg.noise_schedule(sigma_max=jnp.exp(params["params"]["log_max_diffusion"]))
        if alg_cfg.learn_max_diffusion
        else alg_cfg.noise_schedule
    )

    return get_schedule, initial_density, beta_fn, noise_schedule


def inner_step_simulate(
    key,
    model_state,
    params,
    samples,
    log_weights,
    sim_tuple,
    markov_kernel_apply,
    sub_traj,
    config,
    smc_settings,
    batchsize_override,
):
    key, key_gen = jax.random.split(key)
    keys = jax.random.split(key, samples.shape[0])
    log_is_weights, aux = sub_traj_is_weights(
        keys,
        samples,
        model_state,
        params,
        sim_tuple,
        sub_traj,
        stop_grad=config.loss != "rev_kl",
        detach_langevin_pisgrad=config.model.get("model_detach_langevin", True),
    )
    model_samples, target_log_probs, model_subtrajectories = aux

    increments = get_lnz_elbo_increment(log_is_weights, log_weights)

    sub_traj_start_point, sub_traj_end_point, sub_traj_idx, sub_traj_length = sub_traj

    key, key_gen = jax.random.split(key_gen)

    use_resampling, use_mcmc = smc_settings

    (log_density_per_step, noise_schedule, total_steps, (langevin_norm_clip)) = (
        sim_tuple
    )

    next_samples, next_log_weights, acceptance_tuple, debug_tuple = (
        update_samples_log_weights(
            samples=model_samples,
            log_is_weights=log_is_weights,
            markov_kernel_apply=markov_kernel_apply,
            log_weights=log_weights,
            step=sub_traj_end_point[0],
            key=key,
            use_reweighting=use_resampling,
            use_resampling=use_resampling,
            resampler=config.resampler,
            use_markov=use_mcmc,
            resample_threshold=config.resample_threshold,
            log_density_per_step=log_density_per_step,
        )
    )

    next_samples = jax.lax.stop_gradient(next_samples)
    if config.use_markov:
        target_log_probs = log_density_per_step(sub_traj_end_point[0], next_samples)[
            :, None
        ]

    if batchsize_override or (not config.buffer.use_buffer):
        # memory saver hack, don't return the sub-trajectories if not needed
        return (
            next_samples,
            next_log_weights,
            increments,
            target_log_probs,
            log_is_weights,
            None,
            debug_tuple,
        )
    else:
        return (
            next_samples,
            next_log_weights,
            increments,
            target_log_probs,
            log_is_weights,
            model_subtrajectories,
            debug_tuple,
        )


def simulate(
    key_gen,
    model_state,
    params,
    target_log_density,
    markov_kernel_apply,
    traj,
    config,
    smc_settings,
    batchsize_override=0,
):
    batch_size = config.batch_size if batchsize_override == 0 else batchsize_override
    key, key_gen = jax.random.split(key_gen)
    log_density_per_step, initial_density, beta_fn, noise_schedule = (
        get_annealing_fn_and_prior(params, config, target_log_density)
    )
    initial_samples = initial_density.sample(seed=key, sample_shape=(batch_size,))

    initial_log_weights = -jnp.log(batch_size) * jnp.ones(batch_size)
    markov_kernel_apply = markov_kernel_apply(log_density_per_step, beta_fn)
    (
        n_sub_traj,
        sub_traj_start_points,
        sub_traj_end_points,
        sub_traj_indices,
        sub_traj_length,
    ) = traj

    key, key_gen = jax.random.split(key_gen)
    sub_traj_keys = jax.random.split(key, n_sub_traj)
    sim_tuple = (
        log_density_per_step,
        noise_schedule,
        config.num_steps,
        (config.langevin_norm_clip,),
    )

    # Define initial state and per step inputs for scan step
    initial_state = (initial_samples, initial_log_weights)
    per_step_inputs = (
        sub_traj_keys,
        sub_traj_start_points,
        sub_traj_end_points,
        sub_traj_indices,
    )

    # the rollout
    def scan_step(state, per_step_input):
        samples, log_weights = state
        key, sub_traj_start_point, sub_traj_end_point, sub_traj_idx = per_step_input
        sub_traj = (
            sub_traj_start_point,
            sub_traj_end_point,
            sub_traj_idx,
            sub_traj_length,
        )
        (
            next_samples,
            next_log_weights,
            increments,
            target_log_probs,
            log_is_weights,
            subtrajectories,
            (pre_resample_logweights,),
        ) = inner_step_simulate(
            key,
            model_state,
            params,
            samples,
            log_weights,
            sim_tuple,
            markov_kernel_apply,
            sub_traj,
            config,
            smc_settings,
            batchsize_override,
        )

        next_state = (next_samples, next_log_weights)
        per_step_output = (
            next_samples,
            increments,
            target_log_probs,
            log_is_weights,
            subtrajectories,
            (
                resampling.log_effective_sample_size(log_weights),
                resampling.log_effective_sample_size(pre_resample_logweights),
                log_weights,
                pre_resample_logweights,
            ),
        )
        return next_state, per_step_output

    # final_state contains final samples & rnds
    final_state, per_step_outputs = jax.lax.scan(
        scan_step, initial_state, per_step_inputs
    )

    (
        samples,
        (lnz_incs, elbo_incs),
        sub_traj_target_log_probs,
        sub_traj_log_is_weights,
        sub_trajs,
        log_ess_per_subtraj,
    ) = per_step_outputs
    lnz, elbo = jnp.sum(lnz_incs), jnp.sum(elbo_incs)

    return (
        jnp.concatenate([jnp.expand_dims(initial_samples, 0), samples], axis=0),
        jnp.concatenate(
            [jnp.ones((1, batch_size)), sub_traj_target_log_probs[:, :, 0]], axis=0
        ),
        (lnz, elbo),
        (final_state, sub_traj_log_is_weights),
        sub_trajs,
        log_ess_per_subtraj,
    )


def sample_and_concat(key, buffer_samples, new_samples):
    half_N = buffer_samples.shape[1]

    # Sample N/2 random indices in [0,N) from new_samples
    random_indices = jax.random.choice(key, 2 * half_N, shape=(half_N,), replace=False)

    # Gather the sampled subset
    sampled_subset = jax.lax.stop_gradient(new_samples[:, random_indices, :])
    combined_samples = jnp.concatenate([buffer_samples, sampled_subset], axis=1)

    return combined_samples


def make_subtrajectory_boundaries(num_steps, n_sub_traj):
    # Compute boundaries of sub-trajectories
    sub_traj_length = num_steps // n_sub_traj
    sub_traj_start_points = jnp.array(
        [[t * sub_traj_length] for t in range(n_sub_traj)]
    )
    sub_traj_end_points = jnp.array(
        [[(t + 1) * sub_traj_length] for t in range(n_sub_traj)]
    )
    sub_traj_indices = jnp.arange(n_sub_traj)
    traj = (
        n_sub_traj,
        sub_traj_start_points,
        sub_traj_end_points,
        sub_traj_indices,
        sub_traj_length,
    )

    return (
        sub_traj_length,
        sub_traj_start_points,
        sub_traj_end_points,
        sub_traj_indices,
        traj,
    )


def scld_trainer(cfg, target):
    # Initialization

    key_gen = jax.random.PRNGKey(cfg.seed)
    dim = target.dim
    alg_cfg = cfg.algorithm

    target_samples = target.sample(
        seed=jax.random.PRNGKey(cfg.seed), sample_shape=(cfg.eval_samples,)
    )

    n_sub_traj, num_transitions = alg_cfg.n_sub_traj, alg_cfg.num_steps + 1
    (
        sub_traj_length,
        sub_traj_start_points,
        sub_traj_end_points,
        sub_traj_indices,
        traj,
    ) = make_subtrajectory_boundaries(alg_cfg.num_steps, alg_cfg.n_sub_traj)
    traj_inference = traj

    if hasattr(alg_cfg, "n_sub_traj_inference"):
        _, _, _, _, traj_inference = make_subtrajectory_boundaries(
            alg_cfg.num_steps, alg_cfg.n_sub_traj_inference
        )

    # make the buffer
    buffer_on_cpu = False
    buffer = build_prioritised_subtraj_buffer(
        dim,
        alg_cfg.n_sub_traj,
        jnp.array(alg_cfg.buffer.max_length_in_batches * alg_cfg.batch_size, dtype=int),
        jnp.array(alg_cfg.buffer.min_length_in_batches * alg_cfg.batch_size, dtype=int),
        sub_traj_length + 1,  # length of one subtraj (including startpoint)
        sample_with_replacement=alg_cfg.buffer.sample_with_replacement,
        prioritized=alg_cfg.buffer.prioritized,
        temperature=(
            alg_cfg.buffer.temperature if hasattr(alg_cfg.buffer, "temperature") else 1
        ),
        on_cpu=buffer_on_cpu,
    )

    # Define the model and initialize it
    model = None
    if alg_cfg.model.get("name", "PISGRADNet") == "PISGRADNet":
        model = PISGRADNet(**alg_cfg.model)
    else:
        model = StateTimeNetwork(**alg_cfg.model)

    key, key_gen = jax.random.split(key_gen)
    params = model.init(
        key,
        jnp.ones([alg_cfg.batch_size, dim]),
        jnp.ones([alg_cfg.batch_size, 1]),
        jnp.ones([alg_cfg.batch_size, dim]),
    )

    # Add additional parameters to learn
    additional_params = {}
    params_to_freeze = []
    learnt_prior_params = []

    # For TB Loss only
    if alg_cfg.loss in ["rev_tb", "fwd_tb"]:
        # not really tested!
        initial_logZ = (
            alg_cfg.init_logZ if not alg_cfg.leak_true_lnZ else alg_cfg.true_lnZ
        )
        additional_params["logZ_deltas"] = jnp.ones(n_sub_traj) * initial_logZ
    elif alg_cfg.loss in ["rev_kl", "fwd_kl"]:
        assert not alg_cfg.buffer.use_buffer

    # Annealing Schedule
    if alg_cfg.annealing_schedule.schedule_type == "learnt":
        additional_params["betas"] = jnp.ones((alg_cfg.num_steps,))
    else:
        params_to_freeze.append("betas")

    # Gaussian parameters for prior
    if hasattr(alg_cfg, "mfvi") and alg_cfg.mfvi.use_mfvi:
        trained_mean, trained_log_var = mfvi.mfvi_trainer(
            alg_cfg.mfvi, cfg.seed, target
        )
        additional_params["prior_log_stds"] = 0.5 * trained_log_var
        additional_params["prior_mean"] = trained_mean
    else:
        additional_params["prior_log_stds"] = jnp.ones((dim,)) * jnp.log(
            alg_cfg.init_std
        )
        additional_params["prior_mean"] = jnp.zeros((dim,))
        if alg_cfg.prior.learn_variance:
            learnt_prior_params.append("prior_log_stds")
        else:
            params_to_freeze.append("prior_log_stds")

        if alg_cfg.prior.learn_mean:
            learnt_prior_params.append("prior_mean")
        else:
            params_to_freeze.append("prior_mean")

    prior_lr = 0 if not hasattr(alg_cfg, "prior") else alg_cfg.prior.lr

    # learning the max diffusion
    additional_params["log_max_diffusion"] = jnp.log(alg_cfg.max_diffusion)

    if alg_cfg.learn_max_diffusion:
        # check that alg_cfg.noise_schedule is a factory
        assert callable(alg_cfg.noise_schedule(sigma_max=0))
        learnt_prior_params.append("log_max_diffusion")
    else:
        # check that alg_cfg.noise_schedule is a noise_schedule
        assert not callable(alg_cfg.noise_schedule(0))
        params_to_freeze.append("log_max_diffusion")

    params["params"] = {**params["params"], **additional_params}

    lr_scheduler = make_lr_scheduler(alg_cfg)

    optimizer = optax.chain(
        (
            optax.clip(alg_cfg.grad_clip) if alg_cfg.grad_clip > 0 else optax.identity()
        ),  # clip gradients 
        optax.masked(
            optax.adam(learning_rate=lr_scheduler),
            mask=flattened_traversal(
                lambda path, _: path[-1]
                not in ["logZ_deltas", "betas", "prior_log_stds", "prior_mean"]
            ),
        ),
        optax.masked(
            optax.adam(learning_rate=alg_cfg.annealing_schedule.schedule_lr),
            mask=flattened_traversal(lambda path, _: path[-1] in ["betas"]),
        ),
        optax.masked(
            optax.adam(learning_rate=prior_lr),
            mask=flattened_traversal(lambda path, _: path[-1] in learnt_prior_params),
        ),
        optax.masked(
            optax.set_to_zero(),
            mask=flattened_traversal(lambda path, _: path[-1] in params_to_freeze),
        ),
    )

    if hasattr(alg_cfg, "gradient_accumulation_steps"):
        # take gradients steps at fixed intervals
        optimizer = optax.MultiSteps(
            optimizer, every_k_schedule=alg_cfg.gradient_accumulation_steps
        )

    model_state = train_state.TrainState.create(
        apply_fn=model.apply, params=params, tx=optimizer
    )

    markov_kernel_by_step = (
        lambda annealing_fn, _: markov_kernel.MarkovTransitionKernel(
            alg_cfg.mcmc, annealing_fn, num_transitions, fixed_linear_beta_schedule=True
        )
    )
    # versions of simulate function
    simulate_short_train_nojit = partial(
        simulate,
        target_log_density=target.log_prob,
        markov_kernel_apply=markov_kernel_by_step,
        traj=traj,
        config=alg_cfg,
        batchsize_override=0,
        smc_settings=(alg_cfg.use_resampling, alg_cfg.use_markov),
    )

    simulate_short_no_smc = jax.jit(
        partial(
            simulate,
            target_log_density=target.log_prob,
            markov_kernel_apply=markov_kernel_by_step,
            traj=traj,
            config=alg_cfg,
            batchsize_override=cfg.eval_samples,
            smc_settings=(False, False),
        ),
        static_argnames=("config"),
    )

    simulate_short_smc = jax.jit(
        partial(
            simulate,
            target_log_density=target.log_prob,
            markov_kernel_apply=markov_kernel_by_step,
            traj=traj_inference,
            config=alg_cfg,
            batchsize_override=cfg.eval_samples,
            smc_settings=(
                alg_cfg.use_resampling_inference,
                alg_cfg.use_markov_inference,
            ),
        ),
        static_argnames=("config"),
    )

    # for tb-loss
    lnz_upd_fn = (
        partial(
            lnZ_update_jensen, lr=alg_cfg.logZ_step_size, batch_size=alg_cfg.batch_size
        )
        if alg_cfg.use_jensen_trick
        else partial(lnZ_update_vanilla, lr=alg_cfg.logZ_step_size)
    )

    sub_traj_loss = get_loss_fn(alg_cfg.loss)

    key, key_gen = jax.random.split(key_gen)

    get_schedule_and_prior_fn = partial(
        get_annealing_fn_and_prior, alg_cfg=alg_cfg, target_log_prob=target.log_prob
    )
    eval_fn = eval_scld(
        simulate_short_no_smc,
        simulate_short_smc,
        get_schedule_and_prior_fn,
        target,
        target_samples,
        cfg,
    )

    logger = {}
    eval_freq = alg_cfg.n_sim * alg_cfg.n_updates_per_sim // cfg.n_evals

    if eval_freq == 0:
        eval_freq = alg_cfg.n_sim - 1

    if alg_cfg.buffer.use_buffer == False:
        """
        This section of code corresponds to the bufferless training algorithm (Algorithm 2)
        of our paper https://arxiv.org/pdf/2412.07081

        We implement the KL and LV losses
        """

        def kl_loss(key, model_state, params):

            (
                sim_samples,
                sub_traj_target_log_probs,
                (lnz_est, elbo_est),
                ((final_samples, final_weights), sub_traj_logrnds),
                sub_trajs,
                _,
            ) = simulate_short_train_nojit(
                key, model_state, params
            )  # final_weights is not log(1/n) iff resampling
            return -elbo_est, None

        def kl_loss_one_subtraj(key, model_state, params):
            # for some reason this impl is more memory efficient
            key1, key2 = jax.random.split(key)
            annealing_fn, initial_density, _, noise_schedule = (
                get_annealing_fn_and_prior(params, alg_cfg, target.log_prob)
            )
            initial_samples = initial_density.sample(
                seed=key1, sample_shape=(alg_cfg.batch_size,)
            )
            sim_tuple = (
                annealing_fn,
                noise_schedule,
                num_transitions,
                (alg_cfg.langevin_norm_clip,),
            )
            return sub_traj_rev_kl(
                jax.random.split(key2, alg_cfg.batch_size),
                initial_samples,
                None,
                model_state,
                params,
                sim_tuple,
                sub_traj_start_points[0],
                sub_traj_end_points[0],
                sub_traj_indices[0],
                sub_traj_length,
            )

        def lv_loss(key, model_state, params):
            (
                sim_samples,
                sub_traj_target_log_probs,
                (lnz_est, elbo_est),
                ((final_samples, final_weights), sub_traj_logrnds),
                sub_trajs,
                _,
            ) = simulate_short_train_nojit(
                key, model_state, params
            )  # final_weights is not log(1/n) iff resampling
            return (
                jnp.clip(sub_traj_logrnds.var(ddof=0, axis=1), -1e7, 1e7).mean(),
                None,
            )

        loss_fn = None

        if alg_cfg.loss == "rev_kl":
            if alg_cfg.n_sub_traj == 1:
                loss_fn = jax.jit(
                    jax.value_and_grad(kl_loss_one_subtraj, 2, has_aux=True)
                )
            else:
                loss_fn = jax.jit(jax.value_and_grad(kl_loss, 2, has_aux=True))
        elif alg_cfg.loss == "rev_lv":
            loss_fn = jax.jit(jax.value_and_grad(lv_loss, 2, has_aux=True))
        else:
            raise NotImplementedError

        for i in range(alg_cfg.n_sim):
            key, key_gen = jax.random.split(key_gen)

            (loss, aux), grads = loss_fn(key, model_state, model_state.params)
            model_state = model_state.apply_gradients(grads=grads)

            if cfg.use_wandb:
                # note for KL, loss is also the negative ELBO
                wandb.log({"stats/n_inner_its": i, "stats/n_sims": i, "loss": loss})

            if i % eval_freq == 0 or i + 1 == alg_cfg.n_sim:
                key, key_gen = jax.random.split(key_gen)
                logger.update(eval_fn(model_state, model_state.params, key, i))
                logger["stats/step"] = i
                print_results(i, logger, cfg)
                print(f"Loss: {loss}")

                if cfg.use_wandb:
                    wandb.log(logger)
    else:
        """
        Training with buffer
        """
        key, key_gen = jax.random.split(key_gen)

        simulate_short_train = jax.jit(
            simulate_short_train_nojit, static_argnames=("config", "smc_settings")
        )
        (
            init_samples,
            sub_traj_target_log_probs,
            (lnz_est, elbo_est),
            (_, log_rnds),
            inital_subtrajs,
            _,
        ) = simulate_short_train(key, model_state, params)

        print(f"ELBO before training: {elbo_est}")
        print(f"lnZ before training: {lnz_est}")
        wandb.log({"ELBO_no_training": elbo_est, "lnZ_no_training": lnz_est})

        buffer_state = buffer.init(inital_subtrajs, log_rnds)

        def sub_traj_loss_short(
            keys,
            samples,
            next_samples,
            model_state,
            params,
            sub_traj_start_points,
            sub_traj_end_points,
            sub_traj_indices,
        ):
            annealing_fn, initial_density, _, noise_schedule = (
                get_annealing_fn_and_prior(params, alg_cfg, target.log_prob)
            )
            sim_tuple = (
                annealing_fn,
                noise_schedule,
                num_transitions,
                (alg_cfg.langevin_norm_clip,),
            )
            return sub_traj_loss(
                keys,
                samples,
                next_samples,
                model_state,
                params,
                sim_tuple,
                sub_traj_start_points,
                sub_traj_end_points,
                sub_traj_indices,
                sub_traj_length,
                per_sample_rnd_fn=per_subtraj_log_is,
                detach_langevin_pisgrad=alg_cfg.get("model_detach_langevin", True),
            )

        loss_fn = jax.vmap(
            jax.value_and_grad(jax.jit(sub_traj_loss_short), 4, has_aux=True),
            in_axes=(0, 0, 0, None, None, 0, 0, 0),
        )

        for i in range(alg_cfg.n_sim):
            key, key_gen = jax.random.split(key_gen)
            (
                sim_samples,
                sub_traj_target_log_probs,
                (lnz_est, elbo_est),
                ((final_samples, final_weights), sub_traj_logrnds),
                sub_trajs,
                _,
            ) = simulate_short_train(
                key, model_state, model_state.params
            )  # final_weights is not log(1/n) iff resampling

            xs = sub_trajs
            logws = sub_traj_logrnds

            buffer_state = buffer.add(xs, logws, buffer_state=buffer_state)
            if alg_cfg.loss in ["rev_tb"]:
                # warning: TB-loss has not been extensively tested
                new_lnZs = jnp.array(
                    [
                        lnz_upd_fn(
                            model_state.params["params"]["logZ_deltas"][subtraj_id],
                            sub_traj_logrnds[subtraj_id],
                        )
                        for subtraj_id in range(n_sub_traj)
                    ]
                )
                model_state.params["params"]["logZ_deltas"] = new_lnZs

            for j in range(alg_cfg.n_updates_per_sim):

                key, key_gen = jax.random.split(key_gen)
                key2, key_gen = jax.random.split(key_gen)
                old_batch, buffer_indices = buffer.sample(
                    key=key,
                    buffer_state=buffer_state,
                    batch_size=alg_cfg.batch_size // 2,
                )
                train_batch = sample_and_concat(key2, old_batch, sub_trajs)

                key, key_gen = jax.random.split(key_gen)
                keys = jax.random.split(
                    key,
                    (
                        n_sub_traj,
                        alg_cfg.batch_size,
                    ),
                )

                (per_sample_loss, (recomputed_logws, _)), grads_all = loss_fn(
                    keys,
                    train_batch,
                    train_batch,
                    model_state,
                    model_state.params,
                    sub_traj_start_points,
                    sub_traj_end_points,
                    sub_traj_indices,
                )

                if alg_cfg.buffer.update_weights:
                    # update buffer weights as in training-with-buffer algorithm
                    logw_update = recomputed_logws[:, : buffer_indices.shape[1], 0]
                    buffer_state = buffer.upd_weights(
                        logw_update, buffer_indices, buffer_state
                    )

                model_state = gradient_step(model_state, grads_all)

                if cfg.use_wandb:
                    wandb.log(
                        {
                            "loss_hist": per_sample_loss,
                            "stats/n_inner_its": alg_cfg.n_updates_per_sim * i + j,
                            "stats/n_sims": i,
                            "loss": jnp.mean(per_sample_loss),
                        }
                    )

            if i % eval_freq == 0 or i + 1 == alg_cfg.n_sim:
                key, key_gen = jax.random.split(key_gen)
                logger.update(eval_fn(model_state, model_state.params, key, i))
                logger["stats/step"] = i

                print_results(i, logger, cfg)
                print(f"Loss: {jnp.mean(per_sample_loss)}")

                if cfg.use_wandb:
                    wandb.log(logger)
