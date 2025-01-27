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

if __name__ == "__main__":
    exps = [
        "gaussian_mixture40_2D",
        "gaussian_mixture40_50D",
        "gaussian_mixture40_200D",
    ]
    exps = ["student_t_mixture_2D", "student_t_mixture_50D", "student_t_mixture_200D"]
    # exp = 'student_t_mixture_200D'
    algs = [
        "mfvi",
        "gmmvi_jax",
        "smc",
        "aft",
        "craft",
        "fab",
        "mcd",
        "ldvi",
        "pis2",
        "dis",
        "dds2",
        "gsb",
    ]
    algo_name = [
        "MFVI",
        "GMMVI",
        "SMC",
        "AFT",
        "CRAFT",
        "FAB",
        "MCD",
        "LDVI",
        "PIS",
        "DIS",
        "DDS",
        "GBS",
    ]
    # algs = ['mfvi', ]
    for exp in exps:
        strings = []
        for j, alg in enumerate(algs):
            # try:
            if True:
                alg = f"{alg}"
                fields = ["KL/elbo_mov_avg", "other/EMC_mov_avg"]
                job_types = [
                    f"{exp}",
                ]

                for i, exp in enumerate(job_types):
                    config["local"]["job_types"] = [[exp]]
                    config["local"]["fields"] = fields
                    config["local"]["groups"] = [alg]

                    data_dict, config_list = wandb2numpy.export_data(config)

                mean = data_dict["local"][fields[0]]
                std = data_dict["local"][fields[0]]
                min = data_dict["local"][fields[0]]
                max = data_dict["local"][fields[0]]

                # plt.scatter(fevals, mean)
                if alg in ["smc", "aft"]:
                    elbo_best_run = data_dict["local"][fields[0]].max(1)
                    idx = data_dict["local"][fields[0]].argmax(1)
                    elbo_mean_result = elbo_best_run.mean()
                    elbo_std_result = elbo_best_run.std()
                    entropy_mean_result = data_dict["local"][fields[1]][idx].mean()
                    entropy_std_result = data_dict["local"][fields[1]][idx].std()
                    # print(f'ALG: {alg}: -ELBO: {-elbo_mean_result} EUBO: {eubo_mean_result}')
                    print(
                        bcolors.WARNING
                        + f"-------------------{alg}-------------------\n"
                        + bcolors.ENDC
                    )
                    alg_name = alg.split("_")[0]
                    strings.append(
                        f"({algo_name[j]}, {entropy_mean_result}) +- ({algo_name[j]}, {entropy_std_result})"
                    )
                    print(
                        bcolors.WARNING
                        + f"({algo_name[j]}, {entropy_mean_result}) +- ({algo_name[j]}, {entropy_std_result})"
                        + bcolors.ENDC
                    )
                else:
                    elbo_vals = data_dict["local"][fields[0]]
                    elbo_vals = np.where(~np.isnan(elbo_vals), elbo_vals, -np.inf)
                    entropy_vals = data_dict["local"][fields[1]]
                    entropy_vals = np.where(~np.isnan(elbo_vals), entropy_vals, -np.inf)
                    # elbo_ma = moving_average(elbo_vals, 5)
                    # entropy_ma = moving_average(entropy_vals, 5)
                    idx = elbo_vals.argmax(1)

                    # Reshape index_array to fit the required shape for take_along_axis
                    index_array_reshaped = idx[:, np.newaxis]

                    # Use take_along_axis to gather elements
                    result = np.take_along_axis(
                        entropy_vals, index_array_reshaped, axis=1
                    )

                    # Remove the extra dimension added by take_along_axis
                    result = result.squeeze(axis=1)

                    entropy_mean_result = result.mean()
                    entropy_std_result = result.std()
                    alg_name = alg.split("_")[0]
                    print(
                        bcolors.WARNING
                        + f"({algo_name[j]}, {entropy_mean_result}) +- ({algo_name[j]}, {entropy_std_result})"
                        + bcolors.ENDC
                    )
                    strings.append(
                        f"({algo_name[j]}, {entropy_mean_result}) +- ({algo_name[j]}, {entropy_std_result})"
                    )

        # except:
        #     pass

        for string in strings:
            print(string)
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
