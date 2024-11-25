#!/usr/bin/env bash

#SBATCH -p exercise-gpu
#SBATCH --gres=gpu:1
#SBATCH --open-mode=append
#SBATCH -o output/ex4_r2s.txt

########################
### Global to shared ###
########################
bin/memCpy --register2shared -s 1024 -i 10000 -g 1 -t 1
bin/memCpy --register2shared -s 1024 -i 10000 -g 1 -t 2
bin/memCpy --register2shared -s 1024 -i 10000 -g 1 -t 4
bin/memCpy --register2shared -s 1024 -i 10000 -g 1 -t 8
bin/memCpy --register2shared -s 1024 -i 10000 -g 1 -t 16
bin/memCpy --register2shared -s 1024 -i 10000 -g 1 -t 32
bin/memCpy --register2shared -s 1024 -i 10000 -g 1 -t 64
bin/memCpy --register2shared -s 1024 -i 10000 -g 1 -t 128
bin/memCpy --register2shared -s 1024 -i 10000 -g 1 -t 256
bin/memCpy --register2shared -s 1024 -i 10000 -g 1 -t 512
bin/memCpy --register2shared -s 1024 -i 10000 -g 1 -t 1024

bin/memCpy --register2shared -s 4084 -i 10000 -g 1 -t 1
bin/memCpy --register2shared -s 4084 -i 10000 -g 1 -t 2
bin/memCpy --register2shared -s 4084 -i 10000 -g 1 -t 4
bin/memCpy --register2shared -s 4084 -i 10000 -g 1 -t 8
bin/memCpy --register2shared -s 4084 -i 10000 -g 1 -t 16
bin/memCpy --register2shared -s 4084 -i 10000 -g 1 -t 32
bin/memCpy --register2shared -s 4084 -i 10000 -g 1 -t 64
bin/memCpy --register2shared -s 4084 -i 10000 -g 1 -t 128
bin/memCpy --register2shared -s 4084 -i 10000 -g 1 -t 256
bin/memCpy --register2shared -s 4084 -i 10000 -g 1 -t 512
bin/memCpy --register2shared -s 4084 -i 10000 -g 1 -t 1024

bin/memCpy --register2shared -s 16192 -i 10000 -g 1 -t 1
bin/memCpy --register2shared -s 16192 -i 10000 -g 1 -t 2
bin/memCpy --register2shared -s 16192 -i 10000 -g 1 -t 4
bin/memCpy --register2shared -s 16192 -i 10000 -g 1 -t 8
bin/memCpy --register2shared -s 16192 -i 10000 -g 1 -t 16
bin/memCpy --register2shared -s 16192 -i 10000 -g 1 -t 32
bin/memCpy --register2shared -s 16192 -i 10000 -g 1 -t 64
bin/memCpy --register2shared -s 16192 -i 10000 -g 1 -t 128
bin/memCpy --register2shared -s 16192 -i 10000 -g 1 -t 256
bin/memCpy --register2shared -s 16192 -i 10000 -g 1 -t 512
bin/memCpy --register2shared -s 16192 -i 10000 -g 1 -t 1024



bin/memCpy --register2shared -s 49152 -i 10000 -g 1 -t 1
bin/memCpy --register2shared -s 49152 -i 10000 -g 1 -t 2
bin/memCpy --register2shared -s 49152 -i 10000 -g 1 -t 4
bin/memCpy --register2shared -s 49152 -i 10000 -g 1 -t 8
bin/memCpy --register2shared -s 49152 -i 10000 -g 1 -t 16
bin/memCpy --register2shared -s 49152 -i 10000 -g 1 -t 32
bin/memCpy --register2shared -s 49152 -i 10000 -g 1 -t 64
bin/memCpy --register2shared -s 49152 -i 10000 -g 1 -t 128
bin/memCpy --register2shared -s 49152 -i 10000 -g 1 -t 256
bin/memCpy --register2shared -s 49152 -i 10000 -g 1 -t 512
bin/memCpy --register2shared -s 49152 -i 10000 -g 1 -t 1024



 
########################
### Shared to global ###
########################
# bin/memCpy --register2shared -s 1024 -i 10000 -g 1 -t 1
# bin/memCpy --register2shared -s 49152 -i 10000 -g 1 -t 64
# bin/memCpy --register2shared -s 49152 -i 10000 -t 1024

###########################
### Shared to registers ###
###########################
# bin/memCpy --register2shared -s 1024 -i 10000 -g 1 -t 128

###########################
### Registers to shared ###
###########################
# bin/memCpy --register2shared -s 1024 -i 10000 -g 1 -t 512
