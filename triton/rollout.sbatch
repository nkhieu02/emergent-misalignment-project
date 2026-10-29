#!/bin/bash
#SBATCH --job-name=em-rollout
#SBATCH --partition=gpu-v100-16g,gpu-v100-32g
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x-%j.out
# Usage (from the project dir): sbatch triton/rollout.sbatch RUN
set -euo pipefail
RUN=$1
ENV=/scratch/work/$USER/envs/em
export HF_HOME=/scratch/work/$USER/hf-cache PYTHONUNBUFFERED=1 PATH=$ENV/bin:$PATH CC=$ENV/bin/gcc
nvidia-smi --query-gpu=name,memory.total --format=csv
python triton/rollout.py --run-dir "runs/$RUN"
