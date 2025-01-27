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
        "config": {
            "_fields.num_temps": {"values": [128]},
            "_fields.use_resampling": {"values": [True]},
            "_fields.craft_batch_size": {
                # '_fields.batch_size': {
                "values": [1000]
            },
            # '_fields.mcmc_config': {
            #     # 'values': ["hmc_num_leapfrog_steps: 10\nhmc_step_config:\n  step_sizes:\n  - 0.2\n  - 0.2\n  step_times:\n  - 0.0\n  - 0.5\nhmc_steps_per_iter: 1\niters: 1\nreport_step: 1\nrwm_step_config:\n  step_sizes:\n  - 0.1\n  - 0.1\n  step_times:\n  - 0.0\n  - 0.5\nrwm_steps_per_iter: 0\n"]
            #     'values': ["hmc_num_leapfrog_steps: 10\nhmc_step_config:\n  step_sizes:\n  - 0.0001\n  - 0.1\n  step_times:\n  - 0.0\n  - 0.5\nhmc_steps_per_iter: 1\niters: 1\nreport_step: 1\nrwm_step_config:\n  step_sizes:\n  - 0.1\n  - 0.1\n  step_times:\n  - 0.0\n  - 0.5\nrwm_steps_per_iter: 0\n"]
            # },
        },
        "output_path": "",
        # 'history_samples': 150
    }
}

if __name__ == "__main__":
    exp = "funnel"
    algs = ["craft"]
    # algs = ['aft']
    for alg in algs:
        try:
            alg = f"{alg}_abl"
            fields = ["metric/ELBO", "stats/nfe"]
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
            if alg in ["smc_abl", "aft_abl"]:
                best_run = data_dict["local"][fields[0]].max(1)
                mean_result = best_run.mean()
                std_result = best_run.std()
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
                vals = data_dict["local"][fields[0]]
                ma = moving_average(vals, 5)
                best_run = ma.max(1)
                mean_result = best_run.mean()
                std_result = best_run.std()
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
