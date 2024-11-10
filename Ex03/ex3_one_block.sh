#!/usr/bin/env bash

#SBATCH -p exercise-gpu
#SBATCH --gres=gpu:1
#SBATCH --open-mode=append
#SBATCH -o ex3_1blocksout.txt

bin/main --global-coalesced -g 1 -t 1024
bin/main --global-coalesced -g 1 -t 512
bin/main --global-coalesced -g 1 -t 256
bin/main --global-coalesced -g 1 -t 128
bin/main --global-coalesced -g 1 -t 64
bin/main --global-coalesced -g 1 -t 32
bin/main --global-coalesced -g 1 -t 16
bin/main --global-coalesced -g 1 -t 8
bin/main --global-coalesced -g 1 -t 4
bin/main --global-coalesced -g 1 -t 2
bin/main --global-coalesced -g 1 -t 1








