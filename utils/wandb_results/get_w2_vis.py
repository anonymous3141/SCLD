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
        #     # '_fields.num_temps': {
        #     '_fields.num_timesteps': {
        #         'values': [128]
        #     },
        # },
        "output_path": "",
        # 'history_samples': 150
    }
}

exp_2_lnZ = {
    "funnel": 0.0,
    "many_well": 164.69568,
    "gmm": 0.0,
    "gmm40": 0.0,
    "stmm": 0.0,
    "bmm": 0.0,
    "nice": 0.0,
}

if __name__ == "__main__":
    algs = ["mfvi", "smc", "aft", "craft", "pis", "mcd", "ldvi", "cmcd"]
    algs = ["gmmvi", "smc", "aft", "craft", "pis", "mcd", "ldvi", "cmcd", "mfvi"]
    algs = ["mfvi"]

    fields = ["discrepancies/sd", "stats/nfe"]
    job_types = [
        "german_credit",
    ]
    SHOW = True

    for alg in algs:
        alg = f"{alg}_f2"
        for i, exp in enumerate(job_types):
            config["local"]["job_types"] = [[exp]]
            config["local"]["fields"] = fields
            config["local"]["groups"] = [alg]

            data_dict, config_list = wandb2numpy.export_data(config)

        # print(alg)

        fevals = np.round(data_dict["local"]["stats/nfe"].mean(0))
        cond = True  # fevals > 0

        # plt.scatter(fevals, mean)
        if alg in ["smc_f2", "aft_f2"]:

            best_run = data_dict["local"][fields[0]]
            mean_result = best_run.mean(0)
            std_result = best_run.std(0)

            plt.errorbar(fevals, mean_result, yerr=std_result, fmt="o")
        else:

            vals = data_dict["local"][fields[0]]
            ma = moving_average(vals, 5)

            mean_result = ma.mean(0)
            std_result = ma.std(0)

            plt.plot(fevals, mean_result)
            plt.fill_between(
                fevals, mean_result - std_result, mean_result + std_result, alpha=0.3
            )
        # plt.yscale('log')
        plt.xscale("log")
        plt.xlabel("nfe")
        plt.ylabel("$\Delta \log Z$")
        plt.grid()
        plt.title(job_types[0])
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
