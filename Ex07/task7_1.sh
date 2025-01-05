#!/usr/bin/env bash

#SBATCH -p exercise-gpu
#SBATCH --gres=gpu:1
#SBATCH -w csg-brook02
#SBATCH --open-mode=truncate
#SBATCH -o output/ex7_1.txt

 
 
for pow in {4..15}
do
    bin/nbody  -s $((2 ** pow)) -report
 
done
 