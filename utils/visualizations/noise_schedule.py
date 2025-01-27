import matplotlib.pyplot as plt
import numpy as np

if __name__ == "__main__":

    num_time_steps = 128

    def cosine_sq_schedule(step, total_steps, sigma_min=0.008, sigma_max=1.0, pow=2):
        t = step / total_steps
        offset = 1 + sigma_min
        return 0.5 * (sigma_max) * np.cos(0.5 * np.pi * (offset - t) / offset) ** pow

    def linear_noise_schedule(step, total_steps, sigma_min=0.01, sigma_max=1.0):
        t = step / total_steps
        return 0.5 * ((1 - t) * sigma_min + t * sigma_max)

    plt.plot(np.arange(128), cosine_sq_schedule(np.arange(128), 128, pow=4))
    plt.plot(np.arange(128), linear_noise_schedule(np.arange(128), 128))
    plt.show()
