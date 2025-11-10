#!/bin/bash

#SBATCH -A nmf@a100                  # set account

#SBATCH -C a100                      # set gpu_p5 partition (8 80GB A100 GPU per node)

#SBATCH --job-name=1xAtrans # Nom du job
#SBATCH --output=/lustre/fswork/projects/rech/nmf/umu89ib/dev/polpostann/translations_v2/mistralai_Mistral-7B-Instruct-v0.2/%j.log # log file path
#SBATCH --error=/lustre/fswork/projects/rech/nmf/umu89ib/dev/polpostann/translations_v2/mistralai_Mistral-7B-Instruct-v0.2/%j.out # log file path

#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1          # number of nodes
#SBATCH --nodes=1                    # number of nodes

#SBATCH --cpus-per-task=12           # number of cores per task for gpu_p5

#SBATCH --hint=nomultithread         # hyperthreading deactivated

#SBATCH --time=20:00:00              # maximum execution time requested (HH:MM:SS)
#SBATCH --qos=qos_gpu_a100-t3        # qos of 20h limit and 128 GPU par job

# Cleans out modules loaded in interactive and inherited by default
module purge

# Gives access to the modules compatible with the gpu_p6 partition
module load arch/h100

# Show shell used
echo "SHELL:"
echo $0

# Load python environement
conda init
conda activate vllm0.11

# show python used
echo "PYTHON:"
echo $(python --version)
echo $(which python)

# show gpus availables
nvidia-smi

cmd="python /lustre/fswork/projects/rech/nmf/umu89ib/dev/polpostann/async_translations.py --nbgpus=1 --reverse_batch_order"

echo "[RUNNING] ${cmd}"
eval "$cmd"


