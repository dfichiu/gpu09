#!/usr/bin/env bash

#SBATCH -p exercise-gpu
#SBATCH --gres=gpu:1
#SBATCH --open-mode=append
#SBATCH -o output/ex4_2.txt

########################
### Global to shared ###
########################
# bin/memCpy --global2shared -s 1024 -i 10000 -g 1 -t 1024
# bin/memCpy --global2shared -s 1024 -i 10000 -t 64
# bin/memCpy --global2shared -s 1024 -i 10000 -t 1024

# bin/memCpy --global2shared -s 49152 -i 10000 -g 1 -t 1
# bin/memCpy --global2shared -s 49152 -i 10000 -g 1 -t 64
# bin/memCpy --global2shared -s 49152 -i 10000 -t 1024

# bin/memCpy --global2shared -s 4084  -i 10000 -g 4 -t 1024

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

######################
### Bank conflicts ###
######################
bin/memCpy --shared2register_conflict -s 1 -i 10 -stride 17 -t 1024
# bin/memCpy --shared2register_conflict -s 49152 -i 10000 -stride 27 -t 1024
# bin/memCpy --shared2register_conflict -s 49152 -i 10000 -stride 15 -t 1024