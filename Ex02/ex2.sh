#!/usr/bin/env bash

#SBATCH -p exercise-gpu
#SBATCH -o ex2_out.txt

bin/nullKernelAsync
