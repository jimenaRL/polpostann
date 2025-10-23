#!/bin/bash

#SBATCH -A nmf@h100                  # set account

#SBATCH -C h100                      # set gpu_p6 partition (80GB H100 GPU)

# We use 1 node with 2 gpus and let vllm manage gpus paralellisation

#SBATCH --nodes=1                    # number of nodes

#SBATCH --cpus-per-task=24           # number of cores per task for gpu_p6 (1/4 of 4-GPUs H100 node)

#SBATCH --hint=nomultithread         # hyperthreading deactivated

#SBATCH --time=02:00:00              # maximum execution time requested (HH:MM:SS)
#SBATCH --qos=qos_gpu_h100-dev

# Summary table: H100 GPU QoS limits
# QoS              :  time | gpus per job | per project | per QoS
# qos_gpu_h100-t3  :   20h |      512 GPU |     512 GPU |
# qos_gpu_h100-t4  :  100h |       16 GPU |      64 GPU | 192 GPU
# qos_gpu_h100-dev :    2h |       32 GPU |      32 GPU | 384 GPU

# Cleans out modules loaded in interactive and inherited by default
module purge

# Gives access to the modules compatible with the gpu_p6 partition
module load arch/h100

# Load python environement
conda init
conda activate vllm0.11

# show python used
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
echo "OUTFOLDER:"
echo "${OUTFOLDER}"


cmd="python ${SCRIPT} \
       --model_params=${MODELPARAMS} \
       --sampling_params=${SAMPLINGPARAMS} \
       --tweets_file=${TWEETSFILE} \
       --tweets_column=english \
       --system_prompt='You are an expert on French politics.' \
       --user_prompt='Please classify the following social media message (published in the weeks leading up to the 2022 French presidential election) according to whether it explicitly expresses the intention of the author to vote or calls for voting for Macron, Mélenchon, or Le Pen in that election, or expresses neither of these votes. Your response should be based solely on the information contained in the message. Do not assume that a message containing only positive opinions about a particular candidate explicitly expresses the intention to vote for that candidate. Do not assume that a message corresponding to the opinions or political positions of a particular candidate necessarily expresses the intention to vote for that candidate. Do not confuse retweets, indirect speech or a quote to another person with the opinion of the author of the message. Be concise and respond only with the last name or the word "None." Here is the message: \${tweet}' \
       --guided_choice='Macron,Mélenchon,Le Pen,None' \
       --outfolder=${OUTFOLDER}/voteintention/multiple/all"
echo "[RUNNING] ${cmd}"

eval "$cmd"
