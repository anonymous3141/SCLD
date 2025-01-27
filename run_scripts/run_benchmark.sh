python run.py -m seed=0,1,2,3 \
algorithm=$1 \
target=$2 \
use_wandb=True \
wandb.entity=denblessing \
wandb.project=ICML_sampling_benchmark \
visualize_samples=False \
+launcher=hk_gpu \
