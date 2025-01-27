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
    exp = "nice_fashion_784D"
    # exp = 'nice_digits_196D'
    exp = "funnel_10D"
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

    algs = [
        "mfvi",
        "smc",
        "aft",
        "craft",
        "fab",
        "mcd",
        "ldvi",
        "cmcd2",
        "pis2",
        "dis",
        "dds2",
        "gsb",
    ]
    algs = ["smc", "aft", "craft"]
    algs = ["mcd"]
    for alg in algs:
        try:
            alg = f"{alg}"
            fields = ["discrepancies/mmd_mov_avg"]
            job_types = [
                f"{exp}",
            ]

            for i, exp in enumerate(job_types):
                config["local"]["job_types"] = [[exp]]
                config["local"]["fields"] = fields
                config["local"]["groups"] = [alg]

                data_dict, config_list = wandb2numpy.export_data(config)

            mean = data_dict["local"][fields[0]].mean(0)
            std = data_dict["local"][fields[0]].std(0)
            min = data_dict["local"][fields[0]].min(0)
            max = data_dict["local"][fields[0]].max(0)

            # plt.scatter(fevals, mean)
            if alg in ["smc", "aft"]:
                best_run = data_dict["local"][fields[0]].min(1)
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
                vals = np.where(~np.isnan(vals), vals, np.inf)
                best_run = vals.min(1)
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
            # print(bcolors.WARNING + f'Error' + bcolors.ENDC)
            print(bcolors.WARNING + f"& N/A" + bcolors.ENDC)
