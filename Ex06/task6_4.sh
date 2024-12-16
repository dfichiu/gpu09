#!/usr/bin/env bash

#SBATCH -p exercise-gpu
#SBATCH --gres=gpu:1
#SBATCH -w csg-brook02
#SBATCH --open-mode=truncate
#SBATCH -o output/ex6_4_optimized.txt
 
# size 64 to 60k
for pow in {7..17}
do
    bin/reduction -s $((2 ** pow)) -t 1024 --optimized --repo
done
 