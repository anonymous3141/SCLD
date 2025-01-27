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
            # '_fields.dim': {
            # # '_fields.num_timesteps': {
            #     'values': [2]
            # },
        },
        "output_path": "",
        # 'history_samples': 80
    }
}

if __name__ == "__main__":
    exp = "nice"
    algs = ["mfvi", "gmmvi", "smc", "aft", "craft", "fab", "dds"]

    algs = ["gmmvi", "smc", "aft", "craft", "fab", "dds"]
    algs = ["mfvi"]
    # algs = ['mcd', 'ldvi', 'cmcd']
    for alg in algs:
        try:
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

                elbo_vals = data_dict["local"][fields[0]]
                eubo_vals = data_dict["local"][fields[2]]

                idx = elbo_vals.argmax(1)

                mean_result = eubo_vals[np.arange(eubo_vals.shape[0]), idx].mean()
                std_result = eubo_vals[np.arange(eubo_vals.shape[0]), idx].std()

                # best_run = data_dict['local'][fields[0]].max(1)
                # mean_result = best_run.mean()
                # std_result = best_run.std()
                print(
                    bcolors.WARNING
                    + f"-------------------{alg}-------------------\n"
                    + bcolors.ENDC
                )
                print(
                    bcolors.WARNING
                    + f"& ${round(mean_result, 3)} \scriptstyle \pm {round(std_result, 3)}$"
                    + bcolors.ENDC
                )
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

                # vals = data_dict['local'][fields[0]]
                # ma = moving_average(vals, 5)
                # best_run = ma.max(1)
                # mean_result = best_run.mean()
                # std_result = best_run.std()
                print(
                    bcolors.WARNING
                    + f"-------------------{alg}-------------------\n"
                    + bcolors.ENDC
                )
                print(
                    bcolors.WARNING
                    + f"& ${round(mean_result, 3)} \scriptstyle \pm {round(std_result, 3)}$"
                    + bcolors.ENDC
                )
        except:
            print(
                bcolors.WARNING
                + f"-------------------{alg}-------------------\n"
                + bcolors.ENDC
            )
            print(bcolors.WARNING + f"Error" + bcolors.ENDC)
