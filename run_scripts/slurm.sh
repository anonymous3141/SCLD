python3 run.py -m algorithm=scld \
target=many_well_brian,gaussian_mixture40 \
algorithm.n_sub_traj=1,2,8 \
algorithm.use_markov_inference=True,False \
algorithm.use_resampling_inference=True,False \
+launcher=hpc \
+algorithm.config_tag=batch_run5