#!/usr/bin/env bash

#SBATCH -p exercise-gpu
#SBATCH --gres=gpu:1
#SBATCH -w csg-brook02
#SBATCH --open-mode=truncate
#SBATCH -o output/ex7_2_SOA.txt

 
for pow in {4..15}
do
    bin/nbody  -s $((2 ** pow)) -report -SOA
 
done

# Task 7.3
# bin/nbody  -s 200000 -t 1024 -report -SOA