import os

import matplotlib.pyplot as plt
import numpy as np
# import matplotlib
# matplotlib.use('TkAgg')  # Choose an appropriate backend
import tikzplotlib
import wandb2numpy
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
    exp = "seeds"
    algs = [
        "mfvi",
        "gmmvi",
        "nfvi",
        "smc",
        "aft",
        "craft",
        "snf",
        "fab",
        "dds",
        "pis",
        "mcd",
        "ldvi",
        "cmcd",
    ]
    # algs = ['gmmvi']
    for alg in algs:
        try:
            alg = f"{alg}_f2"
            fields = ["metric/ELBO", "stats/nfe", "stats/wallclock"]
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
            if alg in ["smc_f2", "aft_f2"]:
                max_elbo_ind = data_dict["local"][fields[0]].argmax(1)

                wallclock = data_dict["local"][fields[2]]
                wallclock = wallclock[np.arange(wallclock.shape[0]), max_elbo_ind]

                mean_result = wallclock.mean()
                std_result = wallclock.std()

                hour = mean_result // 3600
                minute = mean_result % 3600 // 60
                second = mean_result % 60

                if hour != 0:
                    output = f"& ${round(hour)}h{round(minute)}m{round(second)}s$"
                elif minute != 0:
                    output = f"& ${round(minute)}m{round(second)}s$"
                else:
                    output = f"& ${round(second)}s$"

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

                print(bcolors.WARNING + output + bcolors.ENDC)

            else:
                vals = data_dict["local"][fields[0]]
                ma = moving_average(vals, 5)
                # best_run = ma.max(1)
                max_elbo_ind = ma.argmax(1)

                wallclock = data_dict["local"][fields[2]]
                wallclock = wallclock[np.arange(wallclock.shape[0]), max_elbo_ind]

                mean_result = wallclock.mean()
                std_result = wallclock.std()

                hour = mean_result // 3600
                minute = mean_result % 3600 // 60
                second = mean_result % 60

                if hour != 0:
                    output = f"& ${round(hour)}h{round(minute)}m{round(second)}s$"
                elif minute != 0:
                    output = f"& ${round(minute)}m{round(second)}s$"
                else:
                    output = f"& ${round(second)}s$"

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
                print(bcolors.WARNING + output + bcolors.ENDC)

        except:
            print(
                bcolors.WARNING
                + f"-------------------{alg}-------------------\n"
                + bcolors.ENDC
            )
            print(bcolors.WARNING + f"Error" + bcolors.ENDC)
