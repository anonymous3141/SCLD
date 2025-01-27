import jax
import jax.numpy as jnp


class Buffer:
    def __init__(self, buffer, buffer_elbos, size, num_buffer_batches, num_steps, dim):
        self.size = size
        self.dim = dim
        self.num_buffer_batches = num_buffer_batches
        self.num_steps = num_steps
        self.buffer = buffer
        self.buffer_elbos = buffer_elbos

    def initialise(self, step, batch_size, trajectories, neg_elbo):
        self.buffer = self.buffer.at[step * batch_size : (step + 1) * batch_size].set(
            trajectories
        )
        self.buffer_elbos = self.buffer_elbos.at[
            step * batch_size : (step + 1) * batch_size
        ].set(neg_elbo)
        return self

    def update(self, trajectories, neg_elbo):
        # Add neg_elbos to buffer, and remove the points with lowest neg_elbo
        self.buffer = jnp.concatenate([self.buffer, trajectories], axis=0)
        self.buffer_elbos = jnp.concatenate([self.buffer_elbos, neg_elbo], axis=0)

        # Sort neg_elbo from small to large, and pick smallest.
        buffer_idx = jnp.argsort(self.buffer_elbos, axis=0)[: self.size]

        self.buffer = self.buffer[buffer_idx]
        self.buffer_elbos = self.buffer_elbos[buffer_idx]

    def sample(self, key, batch_size):
        buffer_idx = jax.random.choice(
            key,
            jnp.arange(self.size),
            (batch_size,),
            p=1e6 - self.buffer_elbos,
            replace=True,
        )
        return self.buffer[buffer_idx], self.buffer_elbos[buffer_idx], buffer_idx

    def update_elbos(self, buffer_idx, new_elbos):
        self.buffer_elbos = self.buffer_elbos.at[buffer_idx].set(new_elbos)
        return self

    def _tree_flatten(self):
        children = (self.buffer, self.buffer_elbos)  # arrays / dynamic values
        aux_data = {
            "size": self.size,
            "num_buffer_batches": self.num_buffer_batches,
            "dim": self.dim,
            "num_steps": self.num_steps,
        }  # static values
        return (children, aux_data)

    @classmethod
    def _tree_unflatten(cls, aux_data, children):
        return cls(*children, **aux_data)
