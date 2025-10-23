# Polpostann

Scripts for tweets annotation using llm models via [vllm](https://docs.vllm.ai/en/latest) python inference library, on a HTC clusters managed with Slurm Linux job scheduler.

## Usage

1. We have one python scripts that take as arguments the llm model parameters, the sampling parameters, a csv file where the tweets to annotate are and the name of the column, as well as the system and the user prompts.     


Example of the slurm file

```#!/bin/bash

#SBATCH -A nmf@h100                  # set account
#SBATCH -C h100                      # set gpu_p6 partition (80GB H100 GPU)
# We use 1 node with 2 gpus and let vllm manage gpus paralellisation,
# the rest of the paramametes (gres and ntasks-per-node) are set as input variable to this slurm script.
# We do this to used the same script with different gpus configs for each llm model.
#SBATCH --nodes=1                    # number of nodes
#SBATCH --cpus-per-task=24           # number of cores per task for gpu_p6 (1/4 of 4-GPUs H100 node)
#SBATCH --hint=nomultithread         # hyperthreading deactivated
#SBATCH --time=02:00:00              # maximum execution time requested (HH:MM:SS)
#SBATCH --qos=qos_gpu_h100-dev

# avoid html call to hugging face hub
export HF_HUB_OFFLINE=1

# Cleans out modules loaded in interactive and inherited by default
module purge
# Gives access to the modules compatible with the gpu_p6 partition
module load arch/h100
# Load python environement
conda init
conda activate vllm0.11

# For environement debugging
echo "SHELL: ${0}"
echo "PYTHON:"
echo $(python --version)
echo $(which python)
nvidia-smi

cmd="annotate_tweets.py ${SCRIPT} \
       --model_params=${MODELPARAMS} \
       --sampling_params=${SAMPLINGPARAMS} \
       --tweets_file=${TWEETSFILE} \
       --tweets_column=${TWEETSCOLUMN} \
       --system_prompt=${SYTEMPROMT} \
       --user_prompt=${USERPROMT} \
       --guided_choice='YES,NO' \
       --outfolder=${OUTFOLDER}"

# For scripting debugging
echo "[RUNNING] ${cmd}"

# Code execution
eval "$cmd"

```


Example of script calling using 2 gpu cards ('tensor_parallel_size' vllm model variable).
Deepseek recommends to avoid adding a system prompt; all instructions should be contained within the user prompt.

```
     python annotate_tweets.py \
        --model_params='{"model": "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B", "guided_decoding_backend": "xgrammar", "seed": 19, "dtype": "half", "gpu_memory_utilization": 0.9, "tensor_parallel_size": 2}' \
        --sampling_params='{"temperature": 0.6, "top_p": 0.95}' \
        --tweets_file=200_sampled_xan_seed_123_fr_en.csv \
        --tweets_column=english \
        --system_prompt='You are an expert in leasure activities/' \
        --user_prompt='Please classify the following social media message according to whether it shows the author of the tweet taste for a leisure activity Be concise and answer only YES or NO. Here is the message: ${tweet}' \
        --guided_choice=YES,NO \
        --logfile=2Xv100DeepSeek-R1-Distill-Qwen-32B.log \
        --outfolder=2Xv100DeepSeek-R1-Distill-Qwen-32B
```
