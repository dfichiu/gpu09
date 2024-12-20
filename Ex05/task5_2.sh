#!/usr/bin/env bash

#SBATCH -p exercise-gpu
#SBATCH --gres=gpu:1
#SBATCH -w csg-brook02
#SBATCH --open-mode=append
#SBATCH -o output/ex5_2_v2.txt


# bin/matMul -s 300 --print-matrix  -t 30
# bin/matMul -s 9000 --print-matrix  -t 30 --no-check
# bin/matMul -s 9000 --print-matrix  --shared -t 30 --no-check

for pow in {2..15}
do
    bin/matMul -s $((2 ** pow)) -t 32 --no-check
done
#for pow in {1..5}
#do
#    bin/matMul -s 1024  -t $((2 ** pow)) --no-check
#done