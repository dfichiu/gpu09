#!/usr/bin/env bash

#SBATCH -p exercise-gpu
#SBATCH --gres=gpu:1
#SBATCH --open-mode=append
#SBATCH -o ex3_stride.txt

bin/main --global-coalesced -g 32 -t 1024 -s 32.768 -i 10000
bin/main --global-stride -g 32 -t 1024 -stride 2 -s 32.768 -i 10000
bin/main --global-stride -g 32 -t 1024 -stride 3 -s 32.768 -i 10000
bin/main --global-stride -g 32 -t 1024 -stride 4 -s 32.768 -i 10000
bin/main --global-stride -g 32 -t 1024 -stride 5 -s 32.768 -i 10000
bin/main --global-stride -g 32 -t 1024 -stride 6 -s 32.768 -i 10000
bin/main --global-stride -g 32 -t 1024 -stride 7 -s 32.768 -i 10000
bin/main --global-stride -g 32 -t 1024 -stride 8 -s 32.768 -i 10000
bin/main --global-stride -g 32 -t 1024 -stride 9 -s 32.768 -i 10000
bin/main --global-stride -g 32 -t 1024 -stride 10 -s 32.768 -i 10000
bin/main --global-stride -g 32 -t 1024 -stride 11 -s 32.768 -i 10000


