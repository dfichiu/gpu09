#!/usr/bin/env bash

#SBATCH -p exercise-gpu
#SBATCH --gres=gpu:1
#SBATCH --open-mode=append
#SBATCH -o ex4_1.txt

bin/memCpy --global2shared -s 1024 -i 10000 -t 1
bin/memCpy --global2shared -s 1024 -i 10000 -t 64
bin/memCpy --global2shared -s 1024 -i 10000 -t 1024


bin/memCpy --global2shared -s 49152 -i 10000 -t 1
bin/memCpy --global2shared -s 49152 -i 10000 -t 64
bin/memCpy --global2shared -s 49152 -i 10000 -t 1024

bin/memCpy --shared2global -s 49152 -i 10000 -t 1
bin/memCpy --shared2global -s 49152 -i 10000 -t 64
bin/memCpy --shared2global -s 49152 -i 10000 -t 1024