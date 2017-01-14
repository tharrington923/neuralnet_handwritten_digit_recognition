#!/bin/bash

#SBATCH -J TRAIN_NETWORK_
#SBATCH -n 16
#SBATCH -t 24:00:00
#SBATCH --mem=16G
#SBATCH --mail-type=ALL

./neuralNet_p 
