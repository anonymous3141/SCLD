import os

import matplotlib.pyplot as plt
import numpy as np
# import tikzplotlib
import wandb2numpy
# import matplotlib
# matplotlib.use('TkAgg')  # Choose an appropriate backend
# import tikzplotlib
from utils.path_utils import project_path
from utils.wandb_results.moving_avg import moving_average


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


config = {
    "local": {
        "entity": "denblessing",
        "project": "VI_benchmark",
        "groups": [],
        "fields": [],
        "runs": ["all"],
        "config": {
            # '_fields.num_temps': {
            # # '_fields.num_timesteps': {
            #     'values': [128]
            # },
            "_fields.dim": {
                # '_fields.num_timesteps': {
                "values": [2]
            },
        },
        "output_path": "",
        "history_samples": 80,
    }
}

if __name__ == "__main__":
    exp = "gmm"
    # algs = ['mfvi', 'gmmvi', 'nfvi', 'smc', 'aft', 'craft', 'snf', 'fab', 'dds', 'pis', 'mcd', 'ldvi', 'cmcd']
    algs = [
        "cmcd",
    ]
    elbo_means = []
    eubo_means = []
    for alg in algs:
        # try:
        alg = f"{alg}_new"
        fields = ["metric/ELBO", "stats/nfe", "metric/EUBO"]
        job_types = [
            f"{exp}",
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
        if alg in ["smc_new", "aft_new"]:
            elbo_best_run = data_dict["local"][fields[0]].max(1)
            idx = data_dict["local"][fields[0]].argmax(1)
            elbo_mean_result = elbo_best_run.mean()
            elbo_std_result = elbo_best_run.std()

            eubo_vals = data_dict["local"][fields[2]]
            eubo_mean_result = eubo_vals[np.arange(eubo_vals.shape[0]), idx].mean()

            # print(f'ALG: {alg}: -ELBO: {-elbo_mean_result} EUBO: {eubo_mean_result}')
            print(
                bcolors.WARNING
                + f"-------------------{alg}-------------------\n"
                + bcolors.ENDC
            )
            print(
                bcolors.WARNING
                + f"ALG: {alg}: -ELBO & EUBO: {-elbo_mean_result} {eubo_mean_result}"
                + bcolors.ENDC
            )
        else:
            elbo_vals = data_dict["local"][fields[0]]
            eubo_vals = data_dict["local"][fields[2]]
            elbo_ma = moving_average(elbo_vals, 5)
            eubo_ma = moving_average(eubo_vals, 5)
            idx = elbo_ma.argmax(1)
            elbo_best_run = elbo_ma.max(1)

            # eubo_best_run = eubo_ma[:, idx]
            eubo_best_run = eubo_ma[np.arange(eubo_ma.shape[0]), idx]

            elbo_mean_result = elbo_best_run.mean()
            elbo_std_result = elbo_best_run.std()

            eubo_mean_result = eubo_best_run.mean()
            eubo_std_result = eubo_best_run.std()

            elbo_means.append(elbo_mean_result)
            eubo_means.append(eubo_mean_result)
            print(
                bcolors.WARNING
                + f"ALG: {alg}: -ELBO & EUBO: {-elbo_mean_result} {eubo_mean_result}"
                + bcolors.ENDC
            )

    # fig, ax = plt.subplots()
    # ax.bar(algs, [-l for l in elbo_means], color='red', edgecolor='black', hatch="/")
    # ax.bar(algs, eubo_means, bottom=[-l for l in elbo_means], color='blue', edgecolor='black', hatch='\\')
    # # ax.bar(algs, eubo_means, color='r')
    # # ax.bar(algs, elbo_means, color='b')
    # # ax.bar(algs, [-l + u for l, u in zip(elbo_means, eubo_means)], color='b')
    # # ax.bar(algs, [-l for l in elbo_means], color='b')
    # plt.yscale('log')
    # tikzplotlib.save(os.path.join(project_path('./figures/'), f"jeffreys_{exp}.tex"))
    # plt.show()
    # print(bcolors.WARNING + f'-------------------{alg}-------------------\n' + bcolors.ENDC)
    # print(
    #     bcolors.WARNING + f"& ${round(elbo_mean_result, 3)} \scriptstyle \pm {round(elbo_std_result, 3)}$" + bcolors.ENDC)
    #     # except:
    #     #     print(bcolors.WARNING + f'-------------------{alg}-------------------\n' + bcolors.ENDC)
    #     #     print(bcolors.WARNING + f'Error' + bcolors.ENDC)
