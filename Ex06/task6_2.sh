#!/usr/bin/env bash

#SBATCH -p exercise-gpu
#SBATCH --gres=gpu:1
#SBATCH -w csg-brook02
#SBATCH --open-mode=truncate
#SBATCH -o output/ex6_2.txt



#bin/reduction --cpu  -s   1000
#bin/reduction --cpu  -s 100000
 
#size 64 to 60k
for pow in {15..20}
do
    bin/reduction -s $((2 ** pow))  --cpu --init
done
 