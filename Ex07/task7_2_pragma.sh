#!/usr/bin/env bash

#SBATCH -p exercise-gpu
#SBATCH --gres=gpu:1
#SBATCH -w csg-brook02
#SBATCH --open-mode=append
#SBATCH -o output/ex7_2_pragma.txt

 
 
    bin/nbody  -s $((2 ** 14)) -report -SOA
 
 
 
