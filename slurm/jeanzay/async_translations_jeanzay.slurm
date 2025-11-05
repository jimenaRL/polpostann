#!/bin/bash

#SBATCH -A nmf@h100                  # set account

#SBATCH -C h100                      # set gpu_p6 partition (80GB H100 GPU)

#SBATCH --job-name=1H100trans # Nom du job
#SBATCH --output=/lustre/fswork/projects/rech/nmf/umu89ib/dev/polpostann/translations/mistralai_Mistral-7B-Instruct-v0.2/%j.log # log file path
#SBATCH --error=/lustre/fswork/projects/rech/nmf/umu89ib/dev/polpostann/translations/mistralai_Mistral-7B-Instruct-v0.2/%j.out # log file path

#SBATCH --gres=gpu:h100:1
#SBATCH --ntasks-per-node=1          # number of nodes
#SBATCH --nodes=1                    # number of nodes

#SBATCH --cpus-per-task=24           # number of cores per task for gpu_p6 (1/4 of 4-GPUs H100 node)

#SBATCH --hint=nomultithread         # hyperthreading deactivated

#SBATCH --time=100:00:00              # maximum execution time requested (HH:MM:SS)
#SBATCH --qos=qos_gpu_h100-t4

# Summary table: H100 GPU QoS limits
# QoS              :  time | gpus per job | per project | per QoS
# qos_gpu_h100-t3  :   20h |      512 GPU |     512 GPU |
# qos_gpu_h100-t4  :  100h |       16 GPU |      64 GPU | 192 GPU
# qos_gpu_h100-dev :    2h |       32 GPU |      32 GPU | 384 GPU

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

cmd="python /lustre/fswork/projects/rech/nmf/umu89ib/dev/polpostann/async_translations.py --nbgpus=1"

echo "[RUNNING] ${cmd}"
eval "$cmd"


