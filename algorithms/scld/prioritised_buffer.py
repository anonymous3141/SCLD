from functools import partial
from typing import Callable, Iterable, NamedTuple, Optional, Protocol, Tuple

import chex
import jax.lax
import jax.numpy as jnp
import numpy as np
from algorithms.fab.utils.jax_util import broadcasted_where
from algorithms.scld.prioritised_buffer_utils import *
from jax._src.scipy.special import logsumexp

"""
This implementation of the prioritized buffer comes with some extra functionality.
By default this lives on GPU, but CPU setting is also provisioned

Single sub-trajectory update and retrieval (instead of all the sub-trajectories) is
also provisioned for
"""


def build_prioritised_subtraj_buffer(
    dim: int,
    n_sub_traj: int,
    max_length: int,
    min_length_to_sample: int,
    length_of_subtraj: int,
    prioritized: bool = True,
    sample_with_replacement: bool = False,
    temperature: float = 1,
    on_cpu: bool = False,
) -> PrioritisedBuffer:
    """
    Create replay buffer for batched sampling and adding of data.

    Args:
        dim: Dimension of x data.
        max_length: Maximum length of the buffer.
        min_length_to_sample: Minimum length of buffer required for sampling.
        sample_with_replacement: Whether to sample with replacement.
        n_sub_traj: number of things

    The `max_length` and `min_sample_length` should be sufficiently long to prevent overfitting
    to the replay data. For example, if `min_sample_length` is equal to the
    sampling batch size, then we may overfit to the first batch of data, as we would update
    on it many times during the start of training.
    """
    assert min_length_to_sample <= max_length

    def init(x: chex.Array, log_w: chex.Array) -> PrioritisedBufferState:
        """
        Initialise the buffer state, by filling it above `min_sample_length`.
        x is array of shape (n_sub_traj, batch_size, length_of_subtraj, dim)
        """
        chex.assert_rank(x, 4)
        chex.assert_shape(x[0][0], (length_of_subtraj, dim))
        n_samples = x.shape[1]
        assert (
            n_samples >= min_length_to_sample
        ), "Buffer requires at least `min_sample_length` samples for init."

        current_index = 0
        is_full = False  # whether the buffer is full
        can_sample = False  # whether the buffer is full enough to begin sampling

        device = None if not on_cpu else jax.devices("cpu")[0]

        buffer_array = (
            np.zeros((n_sub_traj, max_length, length_of_subtraj, dim))
            if on_cpu
            else jnp.zeros((n_sub_traj, max_length, length_of_subtraj, dim))
        )

        data = Data(
            x=buffer_array,
            log_w=-jnp.ones(
                (
                    n_sub_traj,
                    max_length,
                ),
                device=device,
            )
            * float("inf"),
        )

        buffer_state = PrioritisedBufferState(
            data=data,
            is_full=is_full,
            can_sample=can_sample,
            current_index=current_index,
        )
        buffer_state = add(x, log_w, buffer_state)
        return buffer_state

    def add(
        x: chex.Array, log_w: chex.Array, buffer_state: PrioritisedBufferState
    ) -> PrioritisedBufferState:
        """Update the buffer's state with a new batch of data."""

        chex.assert_rank(x, 4)
        chex.assert_equal_shape((x[0, 0], buffer_state.data.x[0, 0]))
        batch_size = x.shape[1]
        valid_samples = jnp.isfinite(log_w) & jnp.all(jnp.isfinite(x), axis=(2, 3))
        indices = (jnp.arange(batch_size) + buffer_state.current_index) % max_length

        x, log_w = jax.tree_map(
            lambda a, b: broadcasted_where(valid_samples, a, b),
            (x, log_w),
            (buffer_state.data.x[:, indices], buffer_state.data.log_w[:, indices]),
        )

        if not on_cpu:
            x = buffer_state.data.x.at[:, indices].set(x)
        else:
            buffer_state.data.x[:, indices] = x
            x = buffer_state.data.x

        log_w = buffer_state.data.log_w.at[:, indices].set(log_w)

        new_index = buffer_state.current_index + batch_size
        is_full = jax.lax.select(
            buffer_state.is_full, buffer_state.is_full, new_index >= max_length
        )
        can_sample = jax.lax.select(
            buffer_state.is_full,
            buffer_state.can_sample,
            new_index >= min_length_to_sample,
        )
        current_index = new_index % max_length

        data = Data(x=x, log_w=log_w)
        state = PrioritisedBufferState(
            data=data,
            current_index=current_index,
            is_full=is_full,
            can_sample=can_sample,
        )
        return state

    def sample(
        key: chex.PRNGKey,
        buffer_state: PrioritisedBufferState,
        batch_size: int,
        subtraj_id: int = None,
    ) -> Tuple[chex.Array, chex.Array]:
        """
        Sample a batch from the buffer in proportion to the log weights.
        Returns:
            x: Samples: Shape is (num_subtraj_per_rollout, batch_size, subtraj_length, problem_dim)
            indices: Indices of samples for their location in the buffer state.

        """
        assert batch_size <= min_length_to_sample, (
            "Min length to sample must be greater than or equal to " "the batch size."
        )
        # Get indices.
        buffer_size = max_length if buffer_state.is_full else buffer_state.current_index

        def logprob_transform(ps):
            # we can apply temperature transforms / de-prioritize the buffer etc
            if not prioritized:
                return jnp.zeros_like(ps)
            return ps / temperature

        def sample_from_probabilities(probabilities, key):
            return jax.random.choice(
                key,
                buffer_size,
                shape=(batch_size,),
                replace=sample_with_replacement,
                p=probabilities,
            )

        sample_fn = jax.vmap(sample_from_probabilities, in_axes=(0, 0))
        if subtraj_id is None:
            probs = jnp.exp(logprob_transform(buffer_state.data.log_w[:, :buffer_size]))
            indices = sample_fn(probs, jax.random.split(key, probs.shape[0]))
            return (
                buffer_state.data.x[jnp.arange(indices.shape[0])[:, None], indices],
                indices,
            )
        else:
            probs = jnp.exp(
                logprob_transform(buffer_state.data.log_w[subtraj_id, :buffer_size])
            )
            indices = sample_fn(probs, jax.random.split(key, probs.shape[0]))

            return (
                jnp.array(buffer_state.data.x[subtraj_id, indices], copy=on_cpu),
                indices,
            )  # can also do jnp.array(..., copy=False)

    def adjust_weights(
        new_logws: chex.Array,
        indices: chex.Array,
        buffer_state: PrioritisedBufferState,
        subtraj_id: int = None,
    ) -> PrioritisedBufferState:

        if subtraj_id is None:
            # retrieve all sub-trajectories
            assert indices.shape == new_logws.shape
            assert new_logws.shape[0] == buffer_state.data.log_w.shape[0]
            new_weights = buffer_state.data.log_w.at[
                jnp.arange(indices.shape[0])[:, None], indices
            ].set(jax.lax.stop_gradient(new_logws))
        else:
            # retrieve one sub-trajectory
            assert new_logws.shape[0] == 1 and len(new_logws.shape) == 2
            new_weights = buffer_state.data.log_w.at[
                jnp.array([[subtraj_id]]), indices[subtraj_id]
            ].set(jax.lax.stop_gradient(new_logws))

        data = Data(x=buffer_state.data.x, log_w=new_weights)
        state = PrioritisedBufferState(
            data=data,
            current_index=buffer_state.current_index,
            is_full=buffer_state.is_full,
            can_sample=buffer_state.can_sample,
        )
        return state

    # we don't really use this:
    def sample_n_batches(
        key: chex.PRNGKey,
        buffer_state: PrioritisedBufferState,
        batch_size: int,
        n_batches: int,
    ) -> Iterable[Tuple[chex.Array, chex.Array, chex.Array]]:
        """Returns dataset with n-batches on the leading axis."""
        x, indices = sample(key, buffer_state, batch_size * n_batches)
        dataset = jax.tree_map(
            lambda x: x.reshape((n_batches, batch_size, *x.shape[1:])), (x, indices)
        )
        return dataset

    return PrioritisedBuffer(
        init=init,
        add=add,
        sample=sample,
        sample_n_batches=sample_n_batches,
        min_lengtht_to_sample=min_length_to_sample,
        max_length=max_length,
        upd_weights=adjust_weights,
    )


if __name__ == "__main__":
    dim = 3
    n_sub_traj = 32
    max_length = 1000
    min_length_to_sample = 10
