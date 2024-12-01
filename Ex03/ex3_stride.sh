#!/usr/bin/env bash

#SBATCH -p exercise-gpu
#SBATCH --gres=gpu:1
#SBATCH --open-mode=truncate
#SBATCH -o ex3_stride.txt

bin/main --global-coalesced -g 1 -t 1024 -s 4096 -i 10000
bin/main --global-stride -g 1 -t 1024 -stride 2  -i 10000
bin/main --global-stride -g 1 -t 1024 -stride 3  -i 10000
bin/main --global-stride -g 1 -t 1024 -stride 4 -i 10000
bin/main --global-stride -g 1 -t 1024 -stride 5  -i 10000
bin/main --global-stride -g 1 -t 1024 -stride 6 -i 10000
bin/main --global-stride -g 1 -t 1024 -stride 7  -i 10000
bin/main --global-stride -g 1 -t 1024 -stride 8  -i 10000
bin/main --global-stride -g 1 -t 1024 -stride 9  -i 10000
bin/main --global-stride -g 1 -t 1024 -stride 10  -i 10000
bin/main --global-stride -g 1 -t 1024 -stride 11  -i 10000


