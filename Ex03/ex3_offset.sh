#!/usr/bin/env bash

#SBATCH -p exercise-gpu
#SBATCH --gres=gpu:1
#SBATCH -w csg-brook01
#SBATCH --open-mode=truncate
#SBATCH -o output/ex3_offset.txt

bin/main --global-coalesced -g 1 -t 1024 -s 512 -i 10000
 
for pow in {0..10}
do
    bin/main --global-offset -g 1 -t 1024 -offset $((2**pow)) -s 512 -i 10000
     
done
