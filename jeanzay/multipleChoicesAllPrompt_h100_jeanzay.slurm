#!/bin/bash

#SBATCH -A nmf@h100                  # set account

#SBATCH -C h100                      # set gpu_p6 partition (80GB H100 GPU)

# We use a single node with several gpus and let vllm manage gpus paralellisation
# with the 'tensor_parallel_size' model parameter.
# The number of gpus used is defined by the paramametes 'gres' and 'ntasks-per-node'
# which are given as input variable when launching this slurm script.
# We do this to used the same script with different gpus configs for each llm model.

#SBATCH --nodes=1                    # number of nodes

#SBATCH --cpus-per-task=24           # number of cores per task for gpu_p6 (1/4 of 4-GPUs H100 node)

#SBATCH --hint=nomultithread         # hyperthreading deactivated

#SBATCH --time=20:00:00              # maximum execution time requested (HH:MM:SS)
#SBATCH --qos=qos_gpu_h100-t3

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

# avoid html call to hugging face hub
export HF_HUB_OFFLINE=1

# Code execution
export SCRIPT=/lustre/fswork/projects/rech/nmf/umu89ib/dev/polpostann/annotate_tweets.py

echo ""
echo "SCRIPT:"
echo "${SCRIPT}"

echo ""
echo "MODELPARAMS:"
echo "${MODELPARAMS}"

echo ""
echo "SAMPLINGPARAMS:"
echo "${SAMPLINGPARAMS}"

echo ""
echo "TWEETSFILE:"
echo "${TWEETSFILE}"

echo ""
echo "TWEETSCOLUMN:"
echo "${TWEETSCOLUMN}"

echo ""
echo "SYSTEMPROMT:"
echo "${SYSTEMPROMT}"

echo ""
echo "USERPROMT:"
echo "${USERPROMT}"

echo ""
echo "CHOICES:"
echo "${CHOICES}"

echo ""
echo "OUTFOLDER:"
echo "${OUTFOLDER}"


cmd="python ${SCRIPT} \
       --model_params=${MODELPARAMS} \
       --sampling_params=${SAMPLINGPARAMS} \
       --tweets_file=${TWEETSFILE} \
       --tweets_column=${TWEETSCOLUMN} \
       --system_prompt=${SYSTEMPROMT} \
       --user_prompt=${USERPROMT} \
       --guided_choice=${CHOICES} \
       --outfolder=${OUTFOLDER}"
echo "[RUNNING] ${cmd}"

eval "$cmd"
