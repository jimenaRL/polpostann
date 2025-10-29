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

if [ ${GPU} = 'h100' ]; then
    export GRES=gpu:h100:1
elif [ ${GPU} = 'a100' ]; then
    export GRES=gpu:1
else
    export GRES=gpu:v100:1
fi

echo "LANGUAGE: ${LANGUAGE}"
echo "TASK: ${TASK}"
echo "SEED: ${SEED}"
echo "GPU: ${GPU}"
echo "GRES: ${GRES}"
echo "CANDIDATE: ${CANDIDATE}"

export MODELPARAMS="'{\"model\": \"mistralai/Mistral-Large-Instruct-2411\", \"tokenizer_mode\": \"mistral\", \"config_format\": \"mistral\", \"load_format\": \"mistral\", \"guided_decoding_backend\": \"xgrammar\", \"seed\": ${SEED}, \"tensor_parallel_size\": 4}'"
export SAMPLINGPARAMS="'{\"temperature\": 0.15, \"seed\": ${SEED}, \"max_tokens\": 256}'"
export NAME=Mistral-Large-Instruct-2411-seed${SEED}

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
    --gres=${GRES} \
    --export=ALL \
    ${SERVER}/annotate_tweets_${GPU}_${SERVER}.slurm
