#!/bin/bash

#SBATCH --job-name="Transformer"
#SBATCH --output=job_%j.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user=benzshawelt@msoe.edu
#SBATCH --partition=dgx
#SBATCH --nodes=1
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-gpu=8
#SBATCH --account=practicum


SCRIPT_NAME="NLP"
CONTAINER="/data/containers/msoe-pytorch-23.05-py3.sif"


PYTHON_FILE="run_trainer.py"
SCRIPT_ARGS=""


## SCRIPT
echo "SBATCH SCRIPT: ${SCRIPT_NAME}"
srun hostname; pwd; date;
srun singularity exec --nv -B /data:/data ${CONTAINER} python3 ${PYTHON_FILE} ${SCRIPT_ARGS}


echo "END: " $SCRIPT_NAME