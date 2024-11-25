#!/usr/bin/env bash

#SBATCH -p exercise-gpu
#SBATCH --gres=gpu:1
#SBATCH --open-mode=append
#SBATCH -o output/ex4_2vary_blocks.txt

########################
### Global to shared ###
########################
 

bin/memCpy --shared2global -s 49152 -i 10000 -g 1 -t 1024
bin/memCpy --shared2global -s 49152 -i 10000 -g 2 -t 1024
bin/memCpy --shared2global -s 49152 -i 10000 -g 4 -t 1024
bin/memCpy --shared2global -s 49152 -i 10000 -g 8 -t 1024
bin/memCpy --shared2global -s 49152 -i 10000 -g 16 -t 1024
bin/memCpy --shared2global -s 49152 -i 10000 -g 32 -t 1024
bin/memCpy --shared2global -s 49152 -i 10000 -g 64 -t 1024
bin/memCpy --shared2global -s 49152 -i 10000 -g 128 -t 1024
bin/memCpy --shared2global -s 49152 -i 10000 -g 256 -t 1024
bin/memCpy --shared2global -s 49152 -i 10000 -g 512 -t 1024
bin/memCpy --shared2global -s 49152 -i 10000 -g 1024 -t 1024



 
########################
### Shared to global ###
########################
# bin/memCpy --shared2global -s 1024 -i 10000 -g 1 -t 1
# bin/memCpy --shared2global -s 49152 -i 10000 -g 1 -t 64
# bin/memCpy --shared2global -s 49152 -i 10000 -t 1024

###########################
### Shared to registers ###
###########################
# bin/memCpy --shared2register -s 1024 -i 10000 -g 1 -t 128

###########################
### Registers to shared ###
###########################
# bin/memCpy --register2shared -s 1024 -i 10000 -g 1 -t 512
