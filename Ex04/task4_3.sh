#!/usr/bin/env bash

#SBATCH -p exercise-gpu
#SBATCH --gres=gpu:1
#SBATCH --open-mode=append
#SBATCH -o output/ex4_3.txt

######################
### Bank conflicts ###
######################
bin/memCpy --shared2register_conflict -s 1 -i 10 -stride 1 -t 1024
bin/memCpy --shared2register_conflict -s 1 -i 10 -stride 2 -t 1024
bin/memCpy --shared2register_conflict -s 1 -i 10 -stride 3 -t 1024
bin/memCpy --shared2register_conflict -s 1 -i 10 -stride 4 -t 1024
bin/memCpy --shared2register_conflict -s 1 -i 10 -stride 5 -t 1024
bin/memCpy --shared2register_conflict -s 1 -i 10 -stride 6 -t 1024
bin/memCpy --shared2register_conflict -s 1 -i 10 -stride 7 -t 1024
bin/memCpy --shared2register_conflict -s 1 -i 10 -stride 8 -t 1024
bin/memCpy --shared2register_conflict -s 1 -i 10 -stride 9 -t 1024
bin/memCpy --shared2register_conflict -s 1 -i 10 -stride 10 -t 1024
bin/memCpy --shared2register_conflict -s 1 -i 10 -stride 11 -t 1024
bin/memCpy --shared2register_conflict -s 1 -i 10 -stride 12 -t 1024
bin/memCpy --shared2register_conflict -s 1 -i 10 -stride 13 -t 1024
bin/memCpy --shared2register_conflict -s 1 -i 10 -stride 14 -t 1024
bin/memCpy --shared2register_conflict -s 1 -i 10 -stride 15 -t 1024
bin/memCpy --shared2register_conflict -s 1 -i 10 -stride 16 -t 1024
bin/memCpy --shared2register_conflict -s 1 -i 10 -stride 17 -t 1024
bin/memCpy --shared2register_conflict -s 1 -i 10 -stride 18 -t 1024
bin/memCpy --shared2register_conflict -s 1 -i 10 -stride 19 -t 1024
bin/memCpy --shared2register_conflict -s 1 -i 10 -stride 20 -t 1024
bin/memCpy --shared2register_conflict -s 1 -i 10 -stride 21 -t 1024
bin/memCpy --shared2register_conflict -s 1 -i 10 -stride 22 -t 1024
bin/memCpy --shared2register_conflict -s 1 -i 10 -stride 23 -t 1024
bin/memCpy --shared2register_conflict -s 1 -i 10 -stride 24 -t 1024
bin/memCpy --shared2register_conflict -s 1 -i 10 -stride 25 -t 1024
bin/memCpy --shared2register_conflict -s 1 -i 10 -stride 26 -t 1024
bin/memCpy --shared2register_conflict -s 1 -i 10 -stride 27 -t 1024
bin/memCpy --shared2register_conflict -s 1 -i 10 -stride 28 -t 1024
bin/memCpy --shared2register_conflict -s 1 -i 10 -stride 29 -t 1024
bin/memCpy --shared2register_conflict -s 1 -i 10 -stride 30 -t 1024
bin/memCpy --shared2register_conflict -s 1 -i 10 -stride 31 -t 1024
bin/memCpy --shared2register_conflict -s 1 -i 10 -stride 32 -t 1024
bin/memCpy --shared2register_conflict -s 1 -i 10 -stride 33 -t 1024
bin/memCpy --shared2register_conflict -s 1 -i 10 -stride 34 -t 1024
bin/memCpy --shared2register_conflict -s 1 -i 10 -stride 35 -t 1024
bin/memCpy --shared2register_conflict -s 1 -i 10 -stride 36 -t 1024
bin/memCpy --shared2register_conflict -s 1 -i 10 -stride 37 -t 1024
bin/memCpy --shared2register_conflict -s 1 -i 10 -stride 38 -t 1024
bin/memCpy --shared2register_conflict -s 1 -i 10 -stride 39 -t 1024
bin/memCpy --shared2register_conflict -s 1 -i 10 -stride 40 -t 1024
bin/memCpy --shared2register_conflict -s 1 -i 10 -stride 41 -t 1024
bin/memCpy --shared2register_conflict -s 1 -i 10 -stride 42 -t 1024
bin/memCpy --shared2register_conflict -s 1 -i 10 -stride 43 -t 1024
bin/memCpy --shared2register_conflict -s 1 -i 10 -stride 44 -t 1024
bin/memCpy --shared2register_conflict -s 1 -i 10 -stride 45 -t 1024
bin/memCpy --shared2register_conflict -s 1 -i 10 -stride 46 -t 1024
bin/memCpy --shared2register_conflict -s 1 -i 10 -stride 47 -t 1024
bin/memCpy --shared2register_conflict -s 1 -i 10 -stride 48 -t 1024
bin/memCpy --shared2register_conflict -s 1 -i 10 -stride 49 -t 1024
bin/memCpy --shared2register_conflict -s 1 -i 10 -stride 50 -t 1024
bin/memCpy --shared2register_conflict -s 1 -i 10 -stride 51 -t 1024
bin/memCpy --shared2register_conflict -s 1 -i 10 -stride 52 -t 1024
bin/memCpy --shared2register_conflict -s 1 -i 10 -stride 53 -t 1024
bin/memCpy --shared2register_conflict -s 1 -i 10 -stride 54 -t 1024
bin/memCpy --shared2register_conflict -s 1 -i 10 -stride 55 -t 1024
bin/memCpy --shared2register_conflict -s 1 -i 10 -stride 56 -t 1024
bin/memCpy --shared2register_conflict -s 1 -i 10 -stride 57 -t 1024
bin/memCpy --shared2register_conflict -s 1 -i 10 -stride 58 -t 1024
bin/memCpy --shared2register_conflict -s 1 -i 10 -stride 59 -t 1024
bin/memCpy --shared2register_conflict -s 1 -i 10 -stride 60 -t 1024
bin/memCpy --shared2register_conflict -s 1 -i 10 -stride 61 -t 1024
bin/memCpy --shared2register_conflict -s 1 -i 10 -stride 62 -t 1024
bin/memCpy --shared2register_conflict -s 1 -i 10 -stride 63 -t 1024
bin/memCpy --shared2register_conflict -s 1 -i 10 -stride 64 -t 1024
