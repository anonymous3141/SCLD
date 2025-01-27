import numpy as np
import wandb2numpy
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
        "funnel_10D",
        "gaussian_mixture40_50D",
        "student_t_mixture_50D",
        "nice_digits_196D",
        "nice_fashion_784D",
    ]
    # exps = ['gaussian_mixture40_50D']
    metrics = ["KL/elbo_mov_avg", "KL/eubo_mov_avg"]

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
    # algs = ['cmcd2', 'pis2', 'dis', 'dds2', 'gsb']
    algs = ["fab"]
    for alg in algs:
        strings = []
        for exp in exps:
            for field in metrics:
                try:
                    alg = f"{alg}"

                    # for i, exp in enumerate(job_types):
                    config["local"]["job_types"] = [[exp]]
                    config["local"]["fields"] = [field]

                    config["local"]["groups"] = [alg]

                    data_dict, config_list = wandb2numpy.export_data(config)

                    if alg in ["smc", "aft"]:
                        print(
                            bcolors.WARNING
                            + f"-------------------{alg}:{field}-------------------\n"
                            + bcolors.ENDC
                        )
                        if "elbo" in field:
                            best_run = data_dict["local"][field].max(1)
                        else:
                            best_run = data_dict["local"][field].min(1)
                        mean_result = best_run.mean()
                        std_result = best_run.std()
                        print(
                            bcolors.WARNING
                            + f"& ${round(mean_result, 3)} \scriptstyle \pm {round(std_result, 3)}$"
                            + bcolors.ENDC
                        )
                        strings.append(
                            f"& ${round(mean_result, 3)} \scriptstyle \pm {round(std_result, 3)}$"
                        )
                    else:
                        print(
                            bcolors.WARNING
                            + f"-------------------{alg}:{field}-------------------\n"
                            + bcolors.ENDC
                        )
                        vals = data_dict["local"][field]
                        if "elbo" in field:
                            vals = np.where(
                                (~np.isnan(vals)) & (vals <= 0), vals, -np.inf
                            )
                            best_run = vals.max(1)
                        else:
                            vals = np.where(
                                (~np.isnan(vals)) & (vals >= 0), vals, np.inf
                            )
                            best_run = vals.min(1)
                        mean_result = best_run.mean()
                        std_result = best_run.std()
                        print(
                            bcolors.WARNING
                            + f"& ${round(mean_result, 3)} \scriptstyle \pm {round(std_result, 3)}$"
                            + bcolors.ENDC
                        )
                        strings.append(
                            f"& ${round(mean_result, 3)} \scriptstyle \pm {round(std_result, 3)}$"
                        )
                except:
                    print(
                        bcolors.WARNING
                        + f"-------------------{alg}-------------------\n"
                        + bcolors.ENDC
                    )
                    # print(bcolors.WARNING + f'Error' + bcolors.ENDC)
                    print(bcolors.WARNING + f"& N/A" + bcolors.ENDC)
                    strings.append("& N/A")

        print(
            bcolors.WARNING
            + f"-------------------{alg}-------------------\n"
            + bcolors.ENDC
        )
        for string in strings:
            print(string)
