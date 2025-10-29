#!/bin/bash

GPU=$1
LANGUAGE=french
TASK=support
SEED=1

if [ ${LANGUAGE} = 'french' ]; then
    export CHOICES="OUI,NON"
elif [ ${LANGUAGE} = 'english' ]; then
    export CHOICES="YES,NO"
else
  echo "Error: CHOICES variable is not french nor english"
  exit 1
fi

if [ ${GPU} = 'h100' ]; then
    export GRES=gpu:h100:1
elif [ ${GPU} = 'a100' ]; then
    export GRES=gpu:1
else
    export GRES=gpu:v100:1
fi

export CHOICES="'Macron,Mélenchon,LePen,None'"

echo "TASK: ${TASK}"
echo "LANGUAGE: ${LANGUAGE}"
echo "CHOICES: ${CHOICES}"
echo "SEED: ${SEED}"
echo "GPU: ${GPU}"
echo "GRES: ${GRES}"

export MODELPARAMS="'{\"model\": \"mistralai/Mistral-Small-24B-Instruct-2501\", \"tokenizer_mode\": \"mistral\", \"config_format\": \"mistral\", \"load_format\": \"mistral\", \"guided_decoding_backend\": \"xgrammar\", \"seed\": ${SEED}, \"tensor_parallel_size\": 1}'"
export SAMPLINGPARAMS="'{\"temperature\": 0.15, \"seed\": ${SEED}, \"max_tokens\": 256}'"
export NAME=Mistral-Small-24B-Instruct-2501-seed${SEED}

export TWEETSFILE=${POLPOSTANNPATH}/enumerated_cleaned_text2annotate_2022-03-27_2022-04-25.csv
export TWEETSCOLUMN=${LANGUAGE}
export SYSTEMPROMT=${POLPOSTANNPATH}/prompts/system/system_prompt_${LANGUAGE}.txt
export USERPROMT=${POLPOSTANNPATH}/prompts/user/user_prompt_${TASK}_multiple_all_${LANGUAGE}.txt
export CHOICES="'Macron,Mélenchon,LePen,None'"

export OUTFOLDER=${POLPOSTANNPATH}/outputs_${SERVER}/${LANGUAGE}/${NAME}/guided/${TASK}/multiple/all

sbatch \
    --job-name=${NAME} \
    --output=${OUTFOLDER}/%j.log  \
    --error=${OUTFOLDER}/%j.out  \
    --ntasks-per-node=1 \
    --gres=${GRES} \
    --export=ALL \
    ${SERVER}/annotate_tweets_${GPU}_${SERVER}.slurm
