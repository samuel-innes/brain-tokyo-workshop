#!/bin/sh
#SBATCH -p edu-biai
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH -N 1
#SBATCH -t 0-01:00

python3 train.py cola_train -n 1 -o backprop --add_bert True