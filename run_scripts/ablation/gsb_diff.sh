python run.py -m \
algorithm=gsb \
algorithm.max_diffusion=1,5,10,30 \
target=$1 \
use_wandb=True \
wandb.entity=denblessing \
wandb.project=ICML_sampling_ablation \
visualize_samples=False \
+launcher=hk_gpu