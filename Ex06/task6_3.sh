#!/usr/bin/env bash

#SBATCH -p exercise-gpu
#SBATCH --gres=gpu:1
#SBATCH -w csg-brook02
#SBATCH --open-mode=truncate
#SBATCH -o output/ex6_3.txt

 
# size 1024 to 1m
for pow in {7..17}
do
    bin/reduction -s $((2 ** pow))  --cpu --init --repo
done
 