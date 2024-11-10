#!/usr/bin/env bash

#SBATCH -p exercise-gpu
#SBATCH --gres=gpu:1
#SBATCH --open-mode=append
#SBATCH -o ex3_offset.txt

bin/main --global-coalesced -g 32 -t 1024 -s 32.768 -i 10000
bin/main --global-offset -g 32 -t 1024 -offset 1 -s 32.768 -i 10000
bin/main --global-offset -g 32 -t 1024 -offset 2 -s 32.768 -i 10000
bin/main --global-offset -g 32 -t 1024 -offset 3 -s 32.768 -i 10000
bin/main --global-offset -g 32 -t 1024 -offset 4 -s 32.768 -i 10000
bin/main --global-offset -g 32 -t 1024 -offset 8 -s 32.768 -i 10000
bin/main --global-offset -g 32 -t 1024 -offset 16 -s 32.768 -i 10000
bin/main --global-offset -g 32 -t 1024 -offset 32 -s 32.768 -i 10000
bin/main --global-offset -g 32 -t 1024 -offset 64 -s 32.768 -i 10000
bin/main --global-offset -g 32 -t 1024 -offset 128 -s 32.768 -i 10000
bin/main --global-offset -g 32 -t 1024 -offset 256 -s 32.768 -i 10000
bin/main --global-offset -g 32 -t 1024 -offset 512 -s 32.768 -i 10000


