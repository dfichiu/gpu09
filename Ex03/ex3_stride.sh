#!/usr/bin/env bash

#SBATCH -p exercise-gpu
#SBATCH --gres=gpu:1
#SBATCH -w csg-brook01
#SBATCH --open-mode=truncate
#SBATCH -o output/ex3_stride.txt

bin/main --global-coalesced -g 1 -t 1024 -s 4096 -i 10000
 
for pow in {2..8}
do
    bin/main --global-stride -g 1 -t 1024 -stride $((2**pow)) -i 10000
done
