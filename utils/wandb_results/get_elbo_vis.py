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
    fields = ["metric/ELBO", "stats/nfe"]
    job_types = [
        "ionosphere",
    ]
    SHOW = True
    USE_MOVING_AVERAGE = True

    for i, exp in enumerate(job_types):
        config["local"]["job_types"] = [[exp]]
        config["local"]["fields"] = fields
        config["local"]["groups"] = [alg]

        data_dict, config_list = wandb2numpy.export_data(config)

    if USE_MOVING_AVERAGE:
        fevals = np.round(data_dict["local"]["stats/nfe"].mean(0))
        vals = data_dict["local"][fields[0]]
        ma = moving_average(vals, 5)
        mean = ma.mean(0)
        std = ma.std(0)
    else:
        fevals = np.round(data_dict["local"]["stats/nfe"].mean(0))
        mean = data_dict["local"][fields[0]].mean(0)
        std = data_dict["local"][fields[0]].std(0)

    cond = fevals > 0

    if alg in ["smc_f2", "aft_f2"]:
        plt.errorbar(fevals, mean, yerr=std, fmt="o")
    else:
        plt.plot(fevals[cond], mean[cond])
        plt.fill_between(
            fevals[cond], (mean - std)[cond], (mean + std)[cond], alpha=0.3
        )
    # plt.yscale('log')
    plt.xscale("log")
    plt.xlabel("nfe")
    plt.ylabel("ELBO")
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
