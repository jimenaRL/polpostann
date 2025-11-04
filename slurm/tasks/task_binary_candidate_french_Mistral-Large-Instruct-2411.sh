#!/bin/bash

LANGUAGE=french    # french or english
TASK=$1        # voteintention, support or criticism
SEED=$2        # 1
GPU=$3        # h100 or a100 over jeanzay, h100 or v100 over in2p3
CANDIDATE=$4   # macron, melenchon or lepen

if [ ${LANGUAGE} = 'french' ]; then
    export CHOICES="OUI,NON"
elif [ ${LANGUAGE} = 'english' ]; then
    export CHOICES="YES,NO"
else
  echo "Error: CHOICES variable is not french nor english"
  exit 1
fi

if [ ${GPU} = 'h100' ]; then
    export GRES=gpu:h100:4
elif [ ${GPU} = 'a100' ]; then
    export GRES=gpu:4
else
    export GRES=gpu:v100:4
fi

echo "TASK: ${TASK}"
echo "LANGUAGE: ${LANGUAGE}"
echo "CANDIDATE: ${CANDIDATE}"
echo "CHOICES: ${CHOICES}"
echo "SEED: ${SEED}"
echo "GPU: ${GPU}"
echo "GRES: ${GRES}"
export MODELPARAMS="'{\"model\": \"mistralai/Mistral-Large-Instruct-2411\", \"tokenizer_mode\": \"mistral\", \"config_format\": \"mistral\", \"load_format\": \"mistral\", \"guided_decoding_backend\": \"xgrammar\", \"seed\": ${SEED}, \"tensor_parallel_size\": 4}'"
export SAMPLINGPARAMS="'{\"temperature\": 0.15, \"seed\": ${SEED}, \"max_tokens\": 256}'"
export NAME=Mistral-Large-Instruct-2411-seed${SEED}


export TWEETSFILE=${POLPOSTANNPATH}/enumerated_cleaned_text2annotate_2022-03-27_2022-04-25.csv
export TWEETSCOLUMN=${LANGUAGE}
export SYSTEMPROMT=${POLPOSTANNPATH}/prompts/system/system_prompt_${LANGUAGE}.txt
export USERPROMT=${POLPOSTANNPATH}/prompts/user/user_prompt_${TASK}_binary_${CANDIDATE}_${LANGUAGE}.txt

export OUTFOLDER=${POLPOSTANNPATH}/outputs_${SERVER}/${LANGUAGE}/${NAME}/guided/${TASK}/binary/${CANDIDATE}

sbatch \
    --job-name=${NAME} \
    --output=${OUTFOLDER}/%j.log  \
    --error=${OUTFOLDER}/%j.out  \
    --ntasks-per-node=1 \
    --gres=${GRES} \
    --export=ALL \
    ${SERVER}/annotate_tweets_${GPU}_${SERVER}.slurm
