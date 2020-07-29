# DREAMER FOR PHYRE

Authors: **Niklas Risse**

Institution: **Bielefeld University**


## What is in this repository?
+ The code to generate data from the phyre simulator that is usable as training data for dreamer
+ The code to train dreamer with phyre data
+ The code to run dreamer training on the CITEC GPU CLUSTER of Bielefeld University
+ The code to generate images of imagined trajectories for phyre data

## How to generate data
The python script `data-generation/data_generation.py` can generate data for a given task template. Specify the template number via command line argument. The script will generate 5 random solving rollouts and 5 random not solving rollouts for each task in the template. Example code for task template 00023:

`python data_generation.py --template 00023`

The execution of the script might take a long time (several hours). After the script has terminated, the created episodes can be found in the folder `data-generation/episodes`. To train dreamer with these episodes, copy the folder `data-generation/episodes` to `logdir/dmc_walker_walk/dreamer/1/episodes`. Then create a folder `logdir/dmc_walker_walk/dreamer/1/test_episodes` and move some of the episodes to this folder. These will be the starting points for the imagined trajectories.

## How to train dreamer with phyre data
First you have to generate and move the data as explained above. Then execute the training script with:

`python phyre_dreamer.py --logdir ./logdir/dmc_walker_walk/dreamer/1 --task dmc_walker_walk --log_images False --log_scalars False`
 
 You might need to delete the file `logdir/dmc_walker_walk/dreamer/1/variables.pkl` after each training run.

## How to train dreamer with phyre data on the CITEC GPU CLUSTER
First you have to generate and move the data as explained above. The replace the name `nrisse` in the cluster scripts `cluster-scripts/phyre_dreamer.sbatch` and `cluster-scripts/phyre_dreamer.sh` with your home directory. The you can execute the training with: 

` sbatch cluster-scripts/phyre_dreamer.sbatch` 

You might need to delete the file `logdir/dmc_walker_walk/dreamer/1/variables.pkl` after each training run.

## How to generate images of imagined trajectories
Images of imagined trajectories are generated automatically during training, starting from random timesteps of the episodes in the directory `logdir/dmc_walker_walk/dreamer/1/test_episodes`. They will be saved in the directory `img`, which will be also created during training.

## Installation

## Contact
If you have a problem or question regarding the code, please contact [Niklas Risse](https://github.com/niklasrisse).
