#!/bin/bash

#SBATCH --job-name=mlp
#SBATCH --output=bert_without_mlp.txt
#SBATCH --ntasks=1
#SBATCH --time=72:00:00
#SBATCH --mem=32000
#SBATCH --partition=students
#SBATCH --cpus-per-task=4
#SBATCH --qos=batch
#SBATCH --gres=gpu

python3 baseline.py