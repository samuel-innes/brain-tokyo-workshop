#!/bin/bash

#PBS -l select=256:ncpus=1:mem=4gb

# set max execution time
#PBS -l walltime=4:00:00

# define the queue
#PBS -q short_cpuQ

# set error and output folders
#PBS -o outputs
#PBS -e outputs


module load mpich-3.2
module load python-3.8.13
source /home/marten.mueller/project/bioai/.venv/bin/activate

python3 /home/marten.mueller/project/bioai/brain-tokyo-workshop/WANNRelease/WANN/wann_train.py -n 256 -p 'p/cola.json' -o 'cola'
# python3 /home/marten.mueller/project/bioai/brain-tokyo-workshop/WANNRelease/WANN/wann_test.py -p 'p/cola.json' -o 'log/cola_result_' -i 'log/cola_best.out'

