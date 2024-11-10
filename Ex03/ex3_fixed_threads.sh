#!/usr/bin/env bash

#SBATCH -p exercise-gpu
#SBATCH --gres=gpu:1
#SBATCH --open-mode=append
#SBATCH -o ex3_2out.txt

bin/main --global-coalesced -g 1 -t 512
bin/main --global-coalesced -g 2 -t 512
bin/main --global-coalesced -g 4 -t 512
bin/main --global-coalesced -g 8 -t 512
bin/main --global-coalesced -g 16 -t 512
bin/main --global-coalesced -g 2 -t 512







