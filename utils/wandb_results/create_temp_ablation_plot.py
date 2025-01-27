import os

import matplotlib.pyplot as plt
import numpy as np
# import matplotlib
# matplotlib.use('TkAgg')  # Choose an appropriate backend
import tikzplotlib
import wandb2numpy
from utils.path_utils import project_path
from utils.wandb_results.moving_avg import moving_average


def get_elbo_eubo(alg, dim, T):
    config = {
        "local": {
            "entity": "denblessing",
            "project": "VI_ablations",
            "groups": [],
            "fields": [],
            "runs": ["all"],
            "config": {
                "_fields.dim": {
                    # '_fields.num_timesteps': {
                    "values": [dim]
                },
                "_fields.T": {
                    # '_fields.num_timesteps': {
                    "values": [T]
                },
            },
            "output_path": "",
            "history_samples": 80,
        }
    }

    fields = ["metric/ELBO", "stats/nfe", "metric/EUBO"]
    job_types = [
        "gmm40",
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
    if alg in ["smc_temperature", "aft_temperature"]:
        elbo_best_run = data_dict["local"][fields[0]].max(1)

        elbo_vals = data_dict["local"][fields[0]]
        eubo_vals = data_dict["local"][fields[2]]

        elbo_mean_result = elbo_best_run.mean()
        elbo_std_result = elbo_best_run.std()

        idx = elbo_vals.argmax(1)
        eubo_mean_result = eubo_vals[np.arange(eubo_vals.shape[0]), idx].mean()
        eubo_std_result = eubo_vals[np.arange(eubo_vals.shape[0]), idx].std()

    else:

        elbo_vals = data_dict["local"][fields[0]]
        eubo_vals = data_dict["local"][fields[2]]

        elbo_ma = moving_average(elbo_vals, 5)
        eubo_ma = moving_average(eubo_vals, 5)

        idx = elbo_ma.argmax(1)
        elbo_best_run = elbo_ma.max(1)
        eubo_best_run = eubo_ma[np.arange(eubo_ma.shape[0]), idx]

        elbo_mean_result = elbo_best_run.mean()
        elbo_std_result = elbo_best_run.std()

        eubo_mean_result = eubo_best_run.mean()
        eubo_std_result = eubo_best_run.std()

    return elbo_mean_result, elbo_std_result, eubo_mean_result, eubo_std_result


if __name__ == "__main__":
    alg = "dds_temperature"
    dim = 2
    temp = [8, 16, 64, 128, 256]

    elbo_means = []
    elbo_stds = []
    eubo_means = []
    eubo_stds = []
    for T in temp:

        elbo_mean_result, elbo_std_result, eubo_mean_result, eubo_std_result = (
            get_elbo_eubo(alg=alg, dim=dim, T=T)
        )

        elbo_means.append(elbo_mean_result)
        elbo_stds.append(elbo_std_result)

        eubo_means.append(eubo_mean_result)
        eubo_stds.append(eubo_std_result)

    #################################################################
    x_tick = np.array([0, 1, 2, 3, 4])
    y_tick = np.array(elbo_means)

    plt.plot(x_tick, y_tick)

    plt.fill_between(
        x_tick, y_tick - np.array(elbo_stds), y_tick + np.array(elbo_stds), alpha=0.3
    )

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
    tikzplotlib.save(
        os.path.join(
            project_path("./figures/temp_ablation/"), f"gmm40_dim{dim}_{alg}_elbo.tex"
        )
    )

    plt.show()
    # plt.clf()
    ###################################################################
