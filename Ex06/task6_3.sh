#!/usr/bin/env bash

#SBATCH -p exercise-gpu
#SBATCH --gres=gpu:1
#SBATCH -w csg-brook02
#SBATCH --open-mode=truncate
#SBATCH -o output/ex6_3.txt



#bin/reduction --cpu  -s   1000
#bin/reduction --cpu  -s 100000
 
#size 64 to 60k
for pow in {10..20}
do
    bin/reduction -s $((2 ** pow)) -t 1024 --cpu
done
 