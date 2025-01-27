import os
from datetime import datetime

import hydra
import jax
import matplotlib
import wandb
from jax import config
from omegaconf import DictConfig, OmegaConf
from utils.helper import flatten_dict, reset_device_memory
from utils.train_selector import get_train_fn


@hydra.main(version_base=None, config_path="configs", config_name="base_conf")
def main(cfg: DictConfig) -> None:
    os.environ["HYDRA_FULL_ERROR"] = "1"
    # Load the chosen algorithm-specific configuration dynamically

    # config.update("jax_enable_x64", True) # HACK! Force float64
    cfg = hydra.utils.instantiate(cfg)

    try:  # Overriding default algorithm parameters with experiment specific parameters
        cfg.algorithm = OmegaConf.merge(
            cfg.algorithm, cfg.target[f"{cfg.algorithm.name}"]
        )
    except Exception as e:
        print(
            f"Failed loading target-specific algorithm parameters. Error message: \n {e}"
        )

    try:  # Overriding default generic algorithm parameters with experiment specific parameters
        cfg.algorithm = OmegaConf.merge(cfg.algorithm, cfg.target.all)
    except Exception as e:
        print(
            f"Failed loading target-specific generic algorithm parameters. Error message: \n {e}"
        )

    target = cfg.target.fn

    if not cfg.wandb.get("name"):
        cfg.wandb.name = f"{cfg.algorithm.name}_{cfg.target.name}_{target.dim}_{datetime.now()}_seed{cfg.seed}"

    if not cfg.visualize_samples:
        matplotlib.use("agg")

    if cfg.use_wandb:
        wandb.init(
            **cfg.wandb,
            group=f"{cfg.algorithm.name}",
            job_type=f"{cfg.target.name}_{target.dim}D",
            config=flatten_dict(
                OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
            ),
        )
    train_fn = get_train_fn(cfg.algorithm.name)

    try:
        if cfg.use_jit:
            train_fn(cfg, target)
        else:
            with jax.disable_jit():
                train_fn(cfg, target)
        if cfg.use_wandb:
            wandb.run.summary["error"] = None
            wandb.finish()

    except Exception as e:
        if cfg.use_wandb:
            wandb.run.summary["error"] = str(e)
            wandb.finish(exit_code=1)
        reset_device_memory()
        # jax.profiler.stop_trace()
        raise e


if __name__ == "__main__":
    # jax.profiler.start_trace("tensorboard_logs/")
    main()
