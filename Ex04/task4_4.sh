#!/usr/bin/env bash

#SBATCH -p exercise-gpu
#SBATCH --gres=gpu:1
#SBATCH --open-mode=append
#SBATCH -o ex4_4.txt
bin/matrix_CPU -n 5
bin/matrix_CPU -n 100
bin/matrix_CPU -n 200
bin/matrix_CPU -n 400
bin/matrix_CPU -n 500
bin/matrix_CPU -n 800
bin/matrix_CPU -n 1000
bin/matrix_CPU -n 1500
bin/matrix_CPU -n 1800

bin/matrix_CPU -n 1900
bin/matrix_CPU -n 2000

bin/matrix_CPU -n 2100
 
bin/matrix_CPU -n 3000
bin/matrix_CPU -n 4000
 
bin/matrix_CPU -n 5000
bin/matrix_CPU -n 5500
bin/matrix_CPU -n 6000

