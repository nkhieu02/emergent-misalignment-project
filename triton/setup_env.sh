#!/bin/bash
# One-time: build the Python env on scratch (home is near its quota). Run on a Triton login node.
set -euo pipefail
ENV=/scratch/work/$USER/envs/em
export MAMBA_ROOT_PREFIX=/scratch/work/$USER/micromamba CONDA_PKGS_DIRS=/scratch/work/$USER/conda-pkgs \
    PIP_CACHE_DIR=/scratch/work/$USER/pip-cache

module load micromamba
[ -x "$ENV/bin/python" ] || micromamba create -y -p "$ENV" -c conda-forge python=3.11
# torch JIT-compiles small kernels at runtime and needs a C compiler; compute nodes have none.
[ -x "$ENV/bin/gcc" ] || micromamba install -y -p "$ENV" -c conda-forge c-compiler
# cu126 wheels still support the V100 (sm_70).
"$ENV/bin/pip" install torch --index-url https://download.pytorch.org/whl/cu126
# Same pins as the course notebook's Setup cell.
"$ENV/bin/pip" install "transformers>=5.10.1,<5.16" "huggingface_hub>=1.5.0,<2.0" \
    "peft==0.20.0" "trl==1.12.0" datasets accelerate numpy pandas
"$ENV/bin/python" -c "import torch, transformers, peft, trl; print('torch', torch.__version__, '| transformers', transformers.__version__, '| peft', peft.__version__, '| trl', trl.__version__)"
