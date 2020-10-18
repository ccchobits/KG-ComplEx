#!/bin/bash
#SBATCH -J model_test
#SBATCH --gres=gpu:1
#SBATCH -p PV100q
#SBATCH -w node09



#python3 ../main/main.py --gpu 7 --dim 32 --bs 4096 --init_lr 0.01 --lr_decay 1.1 --bern False  --epochs 1000 --reg 0.000

#python3 ../main/main.py --gpu 7 --dim 64 --bs 4096 --init_lr 0.01 --lr_decay 1.1 --bern False  --epochs 1000 --reg 0.000

#python3 ../main/main.py --gpu 7 --dim 128 --bs 4096 --init_lr 0.01 --lr_decay 1.1 --bern False  --epochs 1000 --reg 0.000

#python3 ../main/main.py --gpu 7 --dim 196 --bs 4096 --init_lr 0.01 --lr_decay 1.1 --bern False  --epochs 1000 --reg 0.000

#python3 ../main/main.py --gpu 7 --dim 64 --bs 4096 --init_lr 0.01 --lr_decay 1.1 --bern False  --epochs 1000 --reg 0.000 --dropout 0.1

#python3 ../main/main.py --gpu 7 --dim 64 --bs 4096 --init_lr 0.01 --lr_decay 1.1 --bern False  --epochs 1000 --reg 0.000 --dropout 0.2

#python3 ../main/main.py --gpu 7 --dim 64 --bs 4096 --init_lr 0.01 --lr_decay 1.1 --bern False  --epochs 1000 --reg 0.000 --dropout 0.3

#python3 ../main/main.py --gpu 7 --dim 64 --bs 4096 --init_lr 0.01 --lr_decay 1.1 --bern False  --epochs 1000 --reg 0.000 --dropout 0.4

#python3 ../main/main.py --gpu 7 --dim 32 --bs 4096 --init_lr 0.01 --lr_decay 1.1 --bern False  --epochs 1000 --reg 0.000 --dropout 0.1

#python3 ../main/main.py --gpu 7 --dim 32 --bs 4096 --init_lr 0.01 --lr_decay 1.1 --bern False  --epochs 1000 --reg 0.000 --dropout 0.15

#python3 ../main/main.py --gpu 7 --dim 32 --bs 4096 --init_lr 0.01 --lr_decay 1.1 --bern False  --epochs 1000 --reg 0.000 --dropout 0.20

#python3 ../main/main.py --gpu 7 --dim 32 --bs 4096 --init_lr 0.01 --lr_decay 1.1 --bern False  --epochs 1000 --reg 0.000 --dropout 0.25

python3 ../main/main.py --gpu 7 --dim 64 --bs 4096 --init_lr 0.01 --lr_decay 1.1 --bern False  --epochs 1000 --reg 0.000 --dropout 0.15

python3 ../main/main.py --gpu 7 --dim 64 --bs 4096 --init_lr 0.01 --lr_decay 1.1 --bern False  --epochs 1000 --reg 0.000 --dropout 0.25




rm "slurm-${SLURM_JOB_ID}.out"
