#!/bin/bash
#SBATCH --job-name=MT_gpt2
#SBATCH --time=1-00:00:00
###SBATCH --gres=gpu:a100:1
###SBATCH --gres=gpu:RTXA6000:1

#SBATCH --ntasks=1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=jackking@mit.edu
#SBATCH --partition=evlab
#SBATCH --mem=100G

source ~/.bashrc

module load openmind8/cuda/11.7
# find the user name
USER_NAME=$(whoami)
unset CUDA_VISIBLE_DEVICES

SU_HOME="/om2/user/${USER_NAME}/superurop"
# run the .bash_profile file from USER_NAME home directory
# . /home/${USER_NAME}/.bash_profile

conda activate modular_transformers
echo $(which python)

python "${SU_HOME}/scripts/generate_toy_datasets.py"

