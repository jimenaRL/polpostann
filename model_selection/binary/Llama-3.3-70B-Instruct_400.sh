#!/bin/bash

LANGUAGE=$1    # frech
TASK=$2        # voteintention
SEED=$3        # 1
GPU=$4         # h100
CANDIDATE=$5   # macron

if [ ${LANGUAGE} = 'french' ]; then
    export CHOICES="OUI,NON"
else
    export CHOICES="YES,NO"
fi

echo "LANGUAGE: ${LANGUAGE}"
echo "TASK: ${TASK}"
echo "SEED: ${SEED}"
echo "GPU: ${GPU}"
echo "CANDIDATE: ${CANDIDATE}"

export MODELPARAMS="'{\"model\": \"meta-llama/Llama-3.3-70B-Instruct\", \"guided_decoding_backend\": \"xgrammar\", \"max_model_len\": 1200, \"seed\": ${SEED}, \"tensor_parallel_size\": 2}'"
export SAMPLINGPARAMS="'{\"seed\": ${SEED}, \"max_tokens\": 256}'"
export NAME=Llama-3.3-70B-Instruct-seed${SEED}

export TWEETSFILE=${POLPOSTANNPATH}/400_balanced_sampled_xan_seed_999_fr_en.csv
export TWEETSCOLUMN=${LANGUAGE}
export SYSTEMPROMT=${POLPOSTANNPATH}/prompts/system/system_prompt_${LANGUAGE}.txt
export USERPROMT=${POLPOSTANNPATH}/prompts/user/user_prompt_${TASK}_binary_${CANDIDATE}_${LANGUAGE}.txt
export CHOICES="OUI,NON"

export OUTFOLDER=${POLPOSTANNPATH}/outputs_${SERVER}/v3ModelSelection${LANGUAGE}/${NAME}/guided/${TASK}/binary/${CANDIDATE}

sbatch \
    --job-name=${NAME} \
    --output=${OUTFOLDER}/%j.log  \
    --error=${OUTFOLDER}/%j.out  \
    --ntasks-per-node=1 \
    --gres=gpu:h100:1 \
    --export=ALL \
    ${SERVER}/annotate_tweets_${GPU}_${SERVER}.slurm
