#!/bin/bash

#PBS -l select=1:ncpus=1:mem=4gb

# set max execution time
#PBS -l walltime=0:05:00

# define the queue
#PBS -q short_cpuQ

# set error and output folders
#PBS -o outputs
#PBS -e outputs


module load mpich-3.2
module load python-3.8.13
source /home/marten.mueller/project/bioai/.venv/bin/activate

# python3 /home/marten.mueller/project/bioai/brain-tokyo-workshop/WANNRelease/WANNTool/train.py cola_train -n 128 -t 4 -o cma --sigma_init 0.5
# python3 /home/marten.mueller/project/bioai/brain-tokyo-workshop/WANNRelease/WANNTool/train.py cola_train -n 128 -t 4
python3 /home/marten.mueller/project/bioai/brain-tokyo-workshop/WANNRelease/WANNTool/train.py cola_train -n 1 -o backprop
# python3 /home/marten.mueller/project/bioai/brain-tokyo-workshop/WANNRelease/WANNTool/model.py cola_train -f log/cola_train.pepg.4.512.best.json