import os

import matplotlib.pyplot as plt
import numpy as np
# import matplotlib
# matplotlib.use('TkAgg')  # Choose an appropriate backend
import tikzplotlib
import wandb2numpy
from utils.path_utils import project_path
from utils.wandb_results.moving_avg import moving_average

config = {
    "local": {
        "entity": "denblessing",
        "project": "VI_benchmark",
        "groups": [],
        "fields": [],
        "runs": ["all"],
        # 'config': {
        #     '_fields.num_temps': {
        #     # '_fields.num_timesteps': {
        #         'values': [128]
        #     },
        # },
        "output_path": "",
        # 'history_samples': 150
    }
}

if __name__ == "__main__":
    alg = "mfvi"
    alg = f"{alg}_f2"
    fields = ["metric/entropy", "metric/ELBO"]  # , 'stats/nfe']
    job_types = [
        "gmm40",
    ]
    SHOW = False
    USE_MOVING_AVERAGE = True

    for i, exp in enumerate(job_types):
        config["local"]["job_types"] = [[exp]]
        config["local"]["fields"] = fields
        config["local"]["groups"] = [alg]

        data_dict, config_list = wandb2numpy.export_data(config)

    if USE_MOVING_AVERAGE:
        kl_mean = -moving_average(data_dict["local"][fields[1]], 5).mean(0)
        kl_std = (-moving_average(data_dict["local"][fields[1]], 5)).std(0)
    else:
        kl_mean = -data_dict["local"][fields[1]].mean(0)
        kl_std = data_dict["local"][fields[1]].std(0)
    # std = data_dict['local'][fields[0]].std(0)[cond]
    # min = data_dict['local'][fields[0]].min(0)[cond]
    # max = data_dict['local'][fields[0]].max(0)[cond]

    idx = np.array([0, kl_mean.shape[0] // 2, kl_mean.shape[0] - 1])

    if alg in ["smc_f2", "aft_f2"]:
        ...
        plt.scatter(kl_mean)
    else:
        # plt.plot(elbo_mean[idx], kl_mean[idx])
        # plt.plot(elbo_mean, kl_mean)
        plt.plot(np.linspace(0, 1, kl_mean.shape[0]), kl_mean)
        plt.fill_between(
            np.linspace(0, 1, kl_mean.shape[0]),
            (kl_mean - kl_std).clip(min=1e-10),
            kl_mean + kl_std,
            alpha=0.3,
        )
    plt.yscale("log")
    plt.xlabel("training progression")
    plt.ylabel("KL")
    # plt.ylim([0,1])
    plt.grid()

    #
    if "/" in fields[0]:
        fields[0] = fields[0].split("/")[-1]
    tikzplotlib.save(
        os.path.join(
            project_path("./figures/"), f"{alg}_{fields[0]}_{job_types[0]}.tex"
        )
    )
    if SHOW:
        plt.show()
