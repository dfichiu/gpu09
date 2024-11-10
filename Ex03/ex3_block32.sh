#!/usr/bin/env bash

#SBATCH -p exercise-gpu
#SBATCH --gres=gpu:1
#SBATCH --open-mode=append
#SBATCH -o ex3_blocks32sout.txt

bin/main --global-coalesced -g 32 -t 1024
bin/main --global-coalesced -g 32 -t 512
bin/main --global-coalesced -g 32 -t 256
bin/main --global-coalesced -g 32 -t 128
bin/main --global-coalesced -g 32 -t 64
bin/main --global-coalesced -g 32 -t 32
bin/main --global-coalesced -g 32 -t 16
bin/main --global-coalesced -g 32 -t 8
bin/main --global-coalesced -g 32 -t 4
bin/main --global-coalesced -g 32 -t 2
bin/main --global-coalesced -g 32 -t 1








