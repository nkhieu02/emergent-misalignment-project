#!/bin/bash
#SBATCH --job-name=em-train
#SBATCH --partition=gpu-v100-16g,gpu-v100-32g
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --output=logs/%x-%j.out
# Usage (from the project dir): sbatch triton/train_sft.sbatch RUN DATA [N_TRAIN] [LR]
set -euo pipefail
RUN=$1 DATA=$2 N_TRAIN=${3:-6000} LR=${4:-2e-4}
ENV=/scratch/work/$USER/envs/em
export HF_HOME=/scratch/work/$USER/hf-cache PYTHONUNBUFFERED=1 PATH=$ENV/bin:$PATH CC=$ENV/bin/gcc
nvidia-smi --query-gpu=name,memory.total --format=csv
python triton/train_sft.py --data "$DATA" --run-dir "runs/$RUN" --n-train "$N_TRAIN" --lr "$LR"
