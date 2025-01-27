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
    exps1 = [
        "gaussian_mixture40_2D",
        "gaussian_mixture40_50D",
        "gaussian_mixture40_200D",
    ]
    exps2 = ["student_t_mixture_2D", "student_t_mixture_50D", "student_t_mixture_200D"]
    metrics = ["logZ/delta_reverse_mov_avg", "logZ/delta_forward_mov_avg"]
    groups = [exps1, exps2]

    # algs = ['mfvi', 'smc', 'aft', 'craft', 'fab', 'mcd', 'ldvi', 'cmcd2', 'pis2', 'dis', 'dds2', 'gsb']
    algs = ["cmcd2", "pis2", "dis", "dds2", "gsb"]
    algs = ["gmmvi_jax"]
    for alg in algs:
        strings = []
        for exps in groups:
            for field in metrics:
                for exp in exps:
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
                            if True:
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
                            if True:
                                vals = data_dict["local"][field]
                                vals = np.where(~np.isnan(vals), vals, np.inf)
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

            print(
                bcolors.WARNING
                + f"-------------------{alg}-------------------\n"
                + bcolors.ENDC
            )
            for string in strings:
                print(string)
