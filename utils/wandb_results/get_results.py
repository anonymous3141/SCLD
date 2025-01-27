import os

import matplotlib.pyplot as plt
import numpy as np
import wandb2numpy
# import matplotlib
# matplotlib.use('TkAgg')  # Choose an appropriate backend
# import tikzplotlib
from utils.path_utils import project_path

config = {
    "local": {
        "entity": "denblessing",
        "project": "VI_benchmark",
        "groups": [],
        "fields": [],
        "runs": ["all"],
        # 'config': {
        #     '_fields.smc_config': {
        #         'values': []
        #     },
        # },
        "output_path": "",
        # 'history_samples': 150
    }
}

if __name__ == "__main__":
    alg = "mfvi_f2"
    # fields = ['divergences/r_div', 'other/num_fevals']
    fields = ["metric/ELBO", "stats/nfe"]
    job_types = [
        "seeds",
    ]

    # if alg in ['gmmvi']:
    if alg in ["mfvi_f2"]:

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
        plt.plot(fevals[cond], mean)
        plt.fill_between(fevals[cond], mean - min, mean + max, alpha=0.3)
        plt.yscale("log")
        plt.xscale("log")
        plt.xlabel("function evaluations")
        plt.ylabel("MMD")
        plt.grid()
        #
        if "/" in fields[0]:
            fields[0] = fields[0].split("/")[-1]
        # tikzplotlib.save(os.path.join(project_path('./figures/'), f"{alg}_{fields[0]}_{job_types[0]}.tex"))
        plt.show()

    elif alg in ["smc"]:
        fevals = []
        res_means = []
        res_stds = []

        for temp in [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]:
            # for temp in [2, 4]:
            config["local"]["job_types"] = [[job_types[0]]]
            config["local"]["fields"] = fields
            config["local"]["groups"] = [alg]
            config["local"]["config"]["_fields.smc_config"]["values"] = [
                f"batch_size: 2000\nnum_temps: {temp}\nresample_threshold: 0.3\nuse_markov: true\nuse_resampling: true\n"
            ]
            data_dict, config_list = wandb2numpy.export_data(config)
            fevals.append(data_dict["local"]["other/num_fevals"].mean(0))
            res_means.append(data_dict["local"][fields[0]].mean(0))
            res_stds.append(data_dict["local"][fields[0]].std(0))

        plt.plot(fevals, res_means)
        plt.fill_between(
            np.array(fevals).reshape(
                -1,
            ),
            (np.array(res_means) - np.array(res_stds)).reshape(
                -1,
            ),
            (np.array(res_means) + np.array(res_stds)).reshape(
                -1,
            ),
            alpha=0.3,
        )
        plt.yscale("log")
        plt.xscale("log")
        plt.xlabel("function evaluations")
        plt.ylabel("MMD")
        plt.grid()
        #
        if "/" in fields[0]:
            fields[0] = fields[0].split("/")[-1]
        tikzplotlib.save(
            os.path.join(
                project_path("./figures/"), f"{alg}_{fields[0]}_{job_types[0]}.tex"
            )
        )
        plt.show()
