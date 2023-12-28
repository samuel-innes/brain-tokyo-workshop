#!/bin/bash

#SBATCH --job-name=mrpc
#SBATCH --output=outputs/mrpc_wann.txt
#SBATCH --ntasks=1
#SBATCH --time=72:00:00
#SBATCH --mem=32000
#SBATCH --partition=students
#SBATCH --cpus-per-task=4
#SBATCH --qos=batch
#SBATCH --gres=gpu

python3 train.py cola_train -n 1 -o backprop --add_bert True --glue_task mrpc