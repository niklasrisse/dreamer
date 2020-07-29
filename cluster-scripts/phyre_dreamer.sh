#!/bin/bash
MUJOCO_GL=osmesa
export PATH="/media/compute/homes/nrisse/.conda/envs/dreamer/bin:$PATH"

python3 /media/compute/homes/nrisse/phyre_dreamer/phyre_dreamer.py --logdir ./logdir/dmc_walker_walk/dreamer/1 --task dmc_walker_walk --log_images False --log_scalars False
