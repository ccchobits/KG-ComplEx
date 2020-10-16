#!/bin/bash
#SBATCH -J model_test
#SBATCH --gres=gpu:1
#SBATCH -p PV100q
#SBATCH -w node09



#python3 ../main/main.py --gpu 2 --dim 32 --n_filter 64 --bs 4096 --init_lr 0.01 --lr_decay 1.1 --bern False --norm 1 --epochs 1000 --reg 0.000 --dropout 0.5 --kernel 1











#rm "slurm-${SLURM_JOB_ID}.out"
