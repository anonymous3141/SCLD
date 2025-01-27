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


exp_2_lnZ = {
    "funnel": 0.0,
    "many_well": 164.69568,
    "gmm40": 0.0,
    "gmm": 0.0,
    "stmm": 0.0,
    "bmm": 0.0,
    "nice": 0.0,
}


def elbo_eubo(alg, dim, batch, exp):
    config = {
        "local": {
            "entity": "denblessing",
            "project": "VI_ablations",
            "groups": [],
            "fields": [],
            "runs": ["all"],
            "config": {
                "_fields.dim": {"values": [dim]},
                "_fields.batch_size": {"values": [batch]},
            },
            "output_path": "",
            # 'history_samples': 80
        }
    }

    fields = ["metric/ELBO", "stats/nfe", "metric/EUBO"]
    job_types = [
        f"{exp}",
    ]

    try:
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

            elbo_mean_result = elbo_best_run.mean()
            elbo_std_result = elbo_best_run.std()

            idx = elbo_vals.argmax(1)
            eubo_mean_result = eubo_vals[np.arange(eubo_vals.shape[0]), idx].mean()
            eubo_std_result = eubo_vals[np.arange(eubo_vals.shape[0]), idx].std()

            print("ELBO result is: ")
            print(
                bcolors.WARNING
                + f"& ${round(elbo_mean_result, 3)} \scriptstyle \pm {round(elbo_std_result, 3)}$"
                + bcolors.ENDC
            )

            print("EUBO result is: ")
            print(
                bcolors.WARNING
                + f"& ${round(eubo_mean_result, 3)} \scriptstyle \pm {round(eubo_std_result, 3)}$"
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

            elbo_mean_result = elbo_best_run.mean()
            elbo_std_result = elbo_best_run.std()

            eubo_mean_result = eubo_best_run.mean()
            eubo_std_result = eubo_best_run.std()

            print("ELBO result is: ")
            print(
                bcolors.WARNING
                + f"& ${round(elbo_mean_result, 3)} \scriptstyle \pm {round(elbo_std_result, 3)}$"
                + bcolors.ENDC
            )

            print("EUBO result is: ")
            print(
                bcolors.WARNING
                + f"& ${round(eubo_mean_result, 3)} \scriptstyle \pm {round(eubo_std_result, 3)}$"
                + bcolors.ENDC
            )

    except:
        print(
            bcolors.WARNING
            + f"-------------------{alg}-------------------\n"
            + bcolors.ENDC
        )
        print(bcolors.WARNING + f"Error" + bcolors.ENDC)


def lnz_revlnz(alg, dim, batch, exp):
    config = {
        "local": {
            "entity": "denblessing",
            "project": "VI_ablations",
            "groups": [],
            "fields": [],
            "runs": ["all"],
            "config": {
                "_fields.dim": {"values": [dim]},
                "_fields.batch_size": {"values": [batch]},
            },
            "output_path": "",
            # 'history_samples': 80
        }
    }

    fields = ["metric/lnZ", "stats/nfe", "metric/rev_lnZ"]
    job_types = [
        f"{exp}",
    ]

    try:
        for i, exp in enumerate(job_types):
            config["local"]["job_types"] = [[exp]]
            config["local"]["fields"] = fields
            config["local"]["groups"] = [alg]

            data_dict, config_list = wandb2numpy.export_data(config)

        delta_ln_Z = np.abs(exp_2_lnZ[exp] - data_dict["local"][fields[0]])

        delta_rev_ln_Z = np.abs(exp_2_lnZ[exp] - data_dict["local"][fields[2]])

        # plt.scatter(fevals, mean)
        if alg in ["smc_new", "aft_new"]:

            lnz_best_run = delta_ln_Z.min(1)

            lnz_mean_result = lnz_best_run.mean()
            lnz_std_result = lnz_best_run.std()

            idx = delta_ln_Z.argmin(1)

            rev_lnz_mean_result = delta_rev_ln_Z[
                np.arange(delta_rev_ln_Z.shape[0]), idx
            ].mean()
            rev_lnz_std_result = delta_rev_ln_Z[
                np.arange(delta_rev_ln_Z.shape[0]), idx
            ].std()

            print("ln_Z result is: ")
            print(
                bcolors.WARNING
                + f"& ${round(lnz_mean_result, 3)} \scriptstyle \pm {round(lnz_std_result, 3)}$"
                + bcolors.ENDC
            )

            print("Rev_ln_Z result is: ")
            print(
                bcolors.WARNING
                + f"& ${round(rev_lnz_mean_result, 3)} \scriptstyle \pm {round(rev_lnz_std_result, 3)}$"
                + bcolors.ENDC
            )

        else:

            lnz_ma = moving_average(delta_ln_Z, 5)
            rev_lnz_ma = moving_average(delta_rev_ln_Z, 5)

            idx = lnz_ma.argmin(1)

            lnz_best_run = lnz_ma.min(1)
            rev_lnz_best_run = rev_lnz_ma[np.arange(rev_lnz_ma.shape[0]), idx]

            lnz_mean_result = lnz_best_run.mean()
            lnz_std_result = lnz_best_run.std()

            rev_lnz_mean_result = rev_lnz_best_run.mean()
            rev_lnz_std_result = rev_lnz_best_run.std()

            print("ln_Z result is: ")
            print(
                bcolors.WARNING
                + f"& ${round(lnz_mean_result, 3)} \scriptstyle \pm {round(lnz_std_result, 3)}$"
                + bcolors.ENDC
            )

            print("Rev_ln_Z result is: ")
            print(
                bcolors.WARNING
                + f"& ${round(rev_lnz_mean_result, 3)} \scriptstyle \pm {round(rev_lnz_std_result, 3)}$"
                + bcolors.ENDC
            )

    except:
        print(
            bcolors.WARNING
            + f"-------------------{alg}-------------------\n"
            + bcolors.ENDC
        )
        print(bcolors.WARNING + f"Error" + bcolors.ENDC)


if __name__ == "__main__":
    exp = "gmm40"
    dim = [2, 50, 200]
    batch = [64, 128, 512, 1024, 2048]
    alg = "fab_new"

    for i in batch:
        for d in dim:

            print("#######################")
            print(alg, f"Batch {i} ", f"Dim {d}")
            print("#######################")

            elbo_eubo(alg=alg, dim=d, batch=i, exp=exp)
            # lnz_revlnz(alg=alg, dim=d, batch=i, exp=exp)
