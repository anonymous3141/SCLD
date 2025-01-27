"""
Controlled Monte Carlo Diffusions (CMCD)
For further details see https://arxiv.org/abs/2307.01050
Code builds on https://github.com/shreyaspadhy/CMCD
"""

from functools import partial
from time import time

import distrax
import jax
import jax.numpy as jnp
import optax
import wandb
from algorithms.cmcd.cmcd_rnd import (log_var, neg_elbo, rnd, traj_bal,
                                      traj_bal_jensens)
from algorithms.common.eval_methods.stochastic_oc_methods import get_eval_fn
from algorithms.common.eval_methods.utils import extract_last_entry
from algorithms.common.models.pisgrad_net import PISGRADNet
from flax.training import train_state
from jax import tree_util
from utils.buffer import Buffer
from utils.helper import inverse_softplus
from utils.print_util import print_results

tree_util.register_pytree_node(Buffer, Buffer._tree_flatten, Buffer._tree_unflatten)


def cmcd2_trainer(cfg, target):
    key_gen = jax.random.PRNGKey(cfg.seed)
    dim = target.dim
    alg_cfg = cfg.algorithm

    # Define initial and target density
    target_samples = target.sample(jax.random.PRNGKey(0), (cfg.eval_samples,))

    # Define the model
    model = PISGRADNet(**alg_cfg.model)
    key, key_gen = jax.random.split(key_gen)
    params = model.init(
        key,
        jnp.ones([alg_cfg.batch_size, dim]),
        jnp.ones([alg_cfg.batch_size, 1]),
        jnp.ones([alg_cfg.batch_size, dim]),
    )

    additional_params = {
        "betas": jnp.ones((alg_cfg.num_steps,)),
        "prior_mean": jnp.zeros((dim,)),
        "prior_std": jnp.ones((dim,)) * inverse_softplus(alg_cfg.init_std),
        "traj_bal": 0.0,
        "ln_z": 0.0,
    }

    params["params"] = {**params["params"], **additional_params}

    def prior_sampler(params, key, n_samples):
        samples = distrax.MultivariateNormalDiag(
            params["params"]["prior_mean"],
            jnp.ones(dim) * jax.nn.softplus(params["params"]["prior_std"]),
        ).sample(seed=key, sample_shape=(n_samples,))
        return samples if alg_cfg.learn_prior else jax.lax.stop_gradient(samples)

    if alg_cfg.learn_prior:

        def prior_log_prob(params, x):
            log_probs = distrax.MultivariateNormalDiag(
                params["params"]["prior_mean"],
                jnp.ones(dim) * jax.nn.softplus(params["params"]["prior_std"]),
            ).log_prob(x)
            return log_probs

    else:

        def prior_log_prob(params, x):
            log_probs = distrax.MultivariateNormalDiag(
                jnp.zeros(dim), jnp.ones(dim) * alg_cfg.init_std
            ).log_prob(x)
            return log_probs

    def get_betas(params):
        b = jax.nn.softplus(params["params"]["betas"])
        b = jnp.cumsum(b) / jnp.sum(b)
        b = b if alg_cfg.learn_betas else jax.lax.stop_gradient(b)

        def get_beta(step):
            return b[jnp.array(step, int)]

        return get_beta

    aux_tuple = (prior_sampler, prior_log_prob, get_betas)

    optimizer = optax.chain(
        optax.zero_nans(),
        optax.clip(alg_cfg.grad_clip),
        optax.adam(learning_rate=alg_cfg.step_size),
    )

    model_state = train_state.TrainState.create(
        apply_fn=model.apply, params=params, tx=optimizer
    )

    if alg_cfg.loss == "log_var":
        loss = jax.jit(
            jax.grad(log_var, 2, has_aux=True),
            static_argnums=(3, 4, 5, 6, 7),
        )
    elif alg_cfg.loss == "elbo":
        loss = jax.jit(
            jax.grad(neg_elbo, 2, has_aux=True), static_argnums=(3, 4, 5, 6, 7)
        )
    elif alg_cfg.loss == "traj_bal":
        loss = jax.jit(
            jax.grad(traj_bal, 2, has_aux=True), static_argnums=(3, 4, 5, 6, 7)
        )
    elif alg_cfg.loss == "traj_bal_jensens":
        loss = jax.jit(
            jax.grad(traj_bal_jensens, 2, has_aux=True),
            static_argnums=(3, 4, 5, 6, 7),
        )
    elif alg_cfg.loss == "traj_bal_jensens_huber":

        def huber_loss(a, delta):
            condition = jnp.abs(a) <= delta
            squared_loss = jnp.square(a)
            linear_loss = 2 * delta * (jnp.abs(a) - 0.5 * delta)
            return jnp.where(condition, squared_loss, linear_loss)

        def pseudo_huber_loss(a, delta):
            # approximates huber(a, delta)
            # Avoids jnp.where so is faster hopefully
            # https://www.explainxkcd.com/wiki/index.php/2295:_Garbage_Math
            return delta * delta * (jnp.sqrt(1 + (a / delta) ** 2) - 1)

        huber_fn = pseudo_huber_loss if alg_cfg.use_pseudo_huber else huber_loss

        loss = jax.jit(
            jax.grad(
                partial(
                    traj_bal_jensens, f_fn=lambda x: huber_fn(x, alg_cfg.huber_delta)
                ),
                2,
                has_aux=True,
            ),
            static_argnums=(3, 4, 5, 6, 7),
        )

    rnd_short = partial(
        rnd,
        batch_size=cfg.eval_samples,
        aux_tuple=aux_tuple,
        target=target,
        num_steps=cfg.algorithm.num_steps,
        noise_schedule=cfg.algorithm.noise_schedule,
        stop_grad=True,
    )

    eval_fn, logger = get_eval_fn(rnd_short, target, target_samples, cfg)

    eval_freq = alg_cfg.iters // cfg.n_evals
    timer = 0

    buffer_size = alg_cfg.buffer.num_buffer_batches * alg_cfg.batch_size
    if alg_cfg.buffer.use_buffer:
        print("Using buffer")
        buffer = Buffer(
            jnp.zeros((buffer_size, alg_cfg.num_steps, dim)),
            jnp.zeros((buffer_size,)),
            buffer_size,
            alg_cfg.buffer.num_buffer_batches,
            alg_cfg.num_steps,
            dim,
        )
    else:
        buffer = None

    ln_z = 0.0
    for step in range(alg_cfg.iters):
        key, key_gen = jax.random.split(key_gen)
        iter_time = time()
        if buffer is None or step < alg_cfg.buffer.num_buffer_batches:
            grads, (neg_elbos, _, trajectories, _, ln_z) = loss(
                key,
                model_state,
                model_state.params,
                alg_cfg.batch_size,
                aux_tuple,
                target,
                alg_cfg.num_steps,
                alg_cfg.noise_schedule,
                None,
                ln_z,
            )
            if buffer is not None:
                # For the first num_buffer_batches, just add trajectories to buffer.
                buffer.initialise(step, alg_cfg.batch_size, trajectories, neg_elbos)
        else:
            grads, (_, _, _, buffer, ln_z) = loss(
                key,
                model_state,
                model_state.params,
                alg_cfg.batch_size // 2,
                aux_tuple,
                target,
                alg_cfg.num_steps,
                alg_cfg.noise_schedule,
                buffer,
                ln_z,
            )
            # print("mean elbo in buffer : ", -jnp.mean(buffer.buffer_elbos))

        timer += time() - iter_time

        model_state = model_state.apply_gradients(grads=grads)

        if (step % eval_freq == 0) or (step == alg_cfg.iters - 1):
            key, key_gen = jax.random.split(key_gen)
            logger["stats/step"].append(step)
            logger["stats/wallclock"].append(timer)
            logger["stats/nfe"].append((step + 1) * alg_cfg.batch_size)
            logger["log_var/traj_bal_ln_z"].append(ln_z)

            logger.update(eval_fn(model_state, key))
            print_results(step, logger, cfg)

            if cfg.use_wandb:
                wandb.log(extract_last_entry(logger))
