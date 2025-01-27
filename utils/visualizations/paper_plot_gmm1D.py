import distrax
import jax
import matplotlib.pyplot as plt
import numpy as jnp

if __name__ == "__main__":
    # Define parameters
    num_components = 3
    means = jnp.array([-5.0, 2.0, 4.0])
    scales = jnp.array([0.5, 1.0, 0.7])
    mixing_probs = jnp.array([0.3, 0.4, 0.3])

    # Create a list of components
    components = [
        distrax.Normal(loc=mean, scale=scale) for mean, scale in zip(means, scales)
    ]

    # Create a single distribution representing the GMM
    target = distrax.MixtureSameFamily(
        distrax.Categorical(probs=mixing_probs),
        distrax.Independent(distrax.Normal(loc=means, scale=scales), 0),
    )

    prior = distrax.Normal(loc=jnp.array([0.0]), scale=jnp.array([1.0]))

    geom_avg = lambda x, alpha: (1 - alpha) * prior.log_prob(
        x
    ) + alpha * target.log_prob(x)
    power = lambda x, alpha: alpha * target.log_prob(x)

    # Generate samples from the GMM
    rng_key = jax.random.PRNGKey(123)
    samples = target.sample(seed=rng_key, sample_shape=(1000,))

    # # Plot histogram of the samples
    # plt.hist(samples, bins=50, density=True, alpha=0.6, color='g')

    # Plot the PDF of the GMM
    x = jnp.linspace(-10, 10, 1000)
    pdf = target.prob(jnp.array(x))
    alpha_vals = jnp.linspace(0, 1, 9)
    fig, axs = plt.subplots(3, 3)
    axs = axs.reshape(-1)
    for i, alpha in enumerate(alpha_vals):
        axs[i].plot(x, jnp.exp(geom_avg(x, alpha)), color="r", linewidth=2)
        axs[i].plot(x, jnp.exp(power(x, alpha)), color="g", linewidth=2)
        axs[i].grid(True)

    # plt.title('1D Gaussian Mixture Model')
    # plt.xlabel('x')
    # plt.ylabel('Probability Density')
    # plt.legend(['PDF', 'Samples'])
    plt.show()
