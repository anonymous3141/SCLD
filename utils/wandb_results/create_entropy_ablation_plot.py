import os

import matplotlib.pyplot as plt
import numpy as np
import tikzplotlib
import wandb2numpy
# import matplotlib
# matplotlib.use('TkAgg')  # Choose an appropriate backend
# import tikzplotlib
from utils.path_utils import project_path
from utils.wandb_results.moving_avg import moving_average


def get_entropy(alg, dim):
    config = {
        "local": {
            "entity": "denblessing",
            "project": "ICML_sampling_benchmark",
            "groups": [],
            "fields": [],
            "runs": ["all"],
            "config": {
                # '_fields.num_temps': {
                # # '_fields.num_timesteps': {
                #     'values': [128]
                # },
                # '_fields.dim': {
                #     # '_fields.num_timesteps': {
                #     'values': [2]
                # },
            },
            "output_path": "",
            # 'history_samples': 60
        }
    }

    fields = ["KL/elbo_mov_avg", "other/EMC_mov_avg"]
    job_types = [
        "gmm",
    ]

    for i, exp in enumerate(job_types):
        config["local"]["job_types"] = [[exp]]
        config["local"]["fields"] = fields
        config["local"]["groups"] = [alg]

        data_dict, config_list = wandb2numpy.export_data(config)

    fevals = data_dict["local"]["stats/nfe"].mean(0)
    cond = fevals > 0
    mean = data_dict["local"][fields[0]].mean(0)[cond]
    std = data_dict["local"][fields[0]].std(0)[cond]
    min = data_dict["local"][fields[0]].min(0)[cond]
    max = data_dict["local"][fields[0]].max(0)[cond]

    # plt.scatter(fevals, mean)
    if alg in ["smc", "aft"]:
        elbo_vals = data_dict["local"][fields[0]]
        eubo_vals = data_dict["local"][fields[2]]

        idx = elbo_vals.argmax(1)

        mean_result = eubo_vals[np.arange(eubo_vals.shape[0]), idx].mean()
        std_result = eubo_vals[np.arange(eubo_vals.shape[0]), idx].std()

    else:

        elbo_vals = data_dict["local"][fields[0]]
        eubo_vals = data_dict["local"][fields[2]]

        elbo_ma = moving_average(elbo_vals, 5)
        eubo_ma = moving_average(eubo_vals, 5)
        idx = elbo_ma.argmax(1)
        elbo_best_run = elbo_ma.max(1)
        eubo_best_run = eubo_ma[np.arange(eubo_ma.shape[0]), idx]

        mean_result = eubo_best_run.mean()
        std_result = eubo_best_run.std()

    return mean_result, std_result


if __name__ == "__main__":
    alg = "cmcd_eps"

    entropy_mean_2, entropy_std_2 = get_entropy(alg, dim=2)
    entropy_mean_50, entropy_std_50 = get_entropy(alg, dim=50)
    entropy_mean_200, entropy_std_200 = get_entropy(alg, dim=200)

    print(entropy_mean_2, entropy_mean_50, entropy_mean_200)

    x_tick = np.array([1, 2, 3])

    y_tick = np.array([entropy_mean_2, entropy_mean_50, entropy_mean_200])

    plt.plot(x_tick, y_tick)

    # if alg in ['smc_f2', 'aft_f2']:
    #     plt.errorbar(fevals, mean, yerr=std, fmt="o")
    # else:
    #     plt.plot(fevals[cond], mean)
    #     plt.fill_between(fevals[cond], mean - std, mean + std, alpha=0.3)
    # plt.yscale('log')

    # plt.xscale('log')
    plt.xlabel("dim")
    plt.ylabel("Entropy")
    plt.grid()
    #
    # if '/' in fields[0]:
    #     fields[0] = fields[0].split('/')[-1]
    # tikzplotlib.save(os.path.join(project_path('./figures/'), f"entropy_{alg}.tex"))

    plt.show()
