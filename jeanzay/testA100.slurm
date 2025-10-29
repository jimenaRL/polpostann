#!/bin/bash

#SBATCH -A nmf@a100                  # set account

#SBATCH -C a100                      # set gpu_p5 partition (8 80GB A100 GPU per node)

# We use a single node with several gpus and let vllm manage gpus paralellisation
# with the 'tensor_parallel_size' model parameter.
# The number of gpus used is defined by the paramametes 'gres' and 'ntasks-per-node'
# which are given as input variable when launching this slurm script.
# We do this to used the same script with different gpus configs for each llm model.

#SBATCH --nodes=1                    # number of nodes
#SBATCH --export=ALL

#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=1           # number of cores per task for gpu_p6 (1/4 of 4-GPUs H100 node)
#SBATCH  --output=%j.log
#SBATCH --error=%j.out

#SBATCH --hint=nomultithread         # hyperthreading deactivated

#SBATCH --time=01:00:00              # maximum execution time requested (HH:MM:SS)
#SBATCH --qos=qos_gpu_a100-t3

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

# avoid html call to hugging face hub
export HF_HUB_OFFLINE=1

