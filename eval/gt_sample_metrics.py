import os

import hydra
import jax.random
import numpy as np
from eval import discrepancies
from omegaconf import DictConfig


class bcolors:
    HEADER = "\033[95m"
    OKBLUE = "\033[94m"
    OKCYAN = "\033[96m"
    OKGREEN = "\033[92m"
    WARNING = "\033[93m"
    FAIL = "\033[91m"
    ENDC = "\033[0m"
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"


import matplotlib.pyplot as plt

# import tikzplotlib


@hydra.main(version_base=None, config_path="../configs", config_name="base_conf")
def main(cfg: DictConfig) -> None:
    os.environ["HYDRA_FULL_ERROR"] = "1"
    # Load the chosen algorithm-specific cfg.targeturation dynamically
    cfg = hydra.utils.instantiate(cfg)

    target = cfg.target.fn
    d = "sd"
    n_samples = 2000

    n_seeds = 20

    discrepancy = np.zeros(n_seeds)

    key, subkey = jax.random.split(jax.random.PRNGKey(1))
    for seed in range(n_seeds):

        groundtruth1 = target.sample(
            seed=jax.random.PRNGKey(0), sample_shape=(n_samples,)
        )  # .clip(min=cfg.target.sample_bounds[0], max=cfg.target.sample_bounds[1])
        key, subkey = jax.random.split(key)
        groundtruth2 = target.sample(
            seed=subkey, sample_shape=(n_samples,)
        )  # .clip(min=cfg.target.sample_bounds[0], max=cfg.target.sample_bounds[1])

        discrepancy[seed] = getattr(discrepancies, f"compute_{d}")(
            gt_samples=groundtruth1, samples=groundtruth2, config=cfg.target
        )
        print(discrepancy[seed])

    print("-------------------")
    # print(discrepancy.mean(0))
    # print(discrepancy.mean(0) + discrepancy.std(0))
    # print(discrepancy.mean(0) - discrepancy.std(0))
    print(
        bcolors.WARNING
        + f"& ${round(discrepancy.mean(0), 3)} \scriptstyle \pm {round(discrepancy.std(0), 3)}$"
        + bcolors.ENDC
    )

    # plt.plot([100, 10e9], np.ones(2) * discrepancy.mean(), c='k')
    # plt.fill_between([100, 10e9], np.ones(2) * (discrepancy.mean() - discrepancy.std()), np.ones(2) * (discrepancy.mean() + discrepancy.std()), color='k', alpha=0.3)
    # plt.yscale('log')
    # plt.xscale('log')
    # plt.xlabel('function evaluations')
    # plt.ylabel(f'{d}')
    # plt.grid()
    # #
    # # tikzplotlib.save(os.path.join(project_path('./figures/'), f"{exp}_{cfg.target.dim}.tex"))
    # plt.show()


if __name__ == "__main__":
    main()
