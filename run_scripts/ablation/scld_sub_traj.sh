python run.py -m seed=0,1,2,3 \
algorithm=$1 \
algorithm.n_sub_traj=1,2,4,8,16,32,64,128 \
target=$2 \
use_wandb=True \
wandb.project=sampling_benchmark \
+launcher=slurm \
