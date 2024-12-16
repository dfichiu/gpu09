#!/usr/bin/env bash

#SBATCH -p exercise-gpu
#SBATCH --gres=gpu:1
#SBATCH -w csg-brook02
#SBATCH --open-mode=truncate
#SBATCH -o output/ex6_6.txt

pow=20

##############################
### Part I: Nsight Systems ###
##############################
# srun nsys profile --trace=cuda -o ./reduction -f true ./bin/reduction -s $((2 ** pow)) -t 1024
# srun nsys profile --trace=cuda -o ./thrust -f true ./bin/reduction -s $((2 ** pow)) -t 1024 --thrust

###############################
### Part II: Nsight Compute ###
###############################
## Default profiling sections
# initial
srun ncu -o ./default-ncu-initial -f ./bin/reduction -s $((2 ** pow)) -t 1024 --init
# # final
srun ncu -o ./default-ncu-final -f ./bin/reduction -s $((2 ** pow)) -t 1024


## All profiling sections
# initial
srun ncu --section "regex:.*" -o ./all-ncu-initial -f ./bin/reduction -s $((2 ** pow)) -t 1024 --init
# # final
srun ncu --section "regex:.*" -o ./all-ncu-final -f ./bin/reduction -s $((2 ** pow)) -t 1024
