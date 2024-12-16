#!/usr/bin/env bash

#SBATCH -p exercise-gpu
#SBATCH --gres=gpu:1
#SBATCH -w csg-brook02
#SBATCH --open-mode=append
#SBATCH -o output/ex5_4.txt


# bin/matMul -s 300 --print-matrix  -t 30
# bin/matMul -s 9000 --print-matrix  -t 30 --no-check
# bin/matMul -s 9000 --print-matrix  --shared -t 30 --no-check


# Task 2: Varying thread number
# for pow in {2..6}
# do
#     bin/matMul -s 2048 -t $((2 ** pow)) --shared --print-header --no-check
# done

# Task 3: Varying problem size
for pow in {5..15}
do
    bin/matMul -s $((2 ** pow)) -t 32 --shared  --no-check
done