LANGUAGE=$1
TASK=$2
SEED=$3
GPU=$4

export MODELPARAMS="'{\"model\": \"mistralai/Mistral-Large-Instruct-2411\", \"tokenizer_mode\": \"mistral\", \"config_format\": \"mistral\", \"load_format\": \"mistral\", \"guided_decoding_backend\": \"xgrammar\", \"seed\": ${SEED}, \"tensor_parallel_size\": 4}'"
export SAMPLINGPARAMS="'{\"temperature\": 0.15, \"seed\": ${SEED}, \"max_tokens\": 256}'"
export NAME=Mistral-Large-Instruct-2411-seed${SEED}

export TWEETSFILE=${POLPOSTANNPATH}/400_balanced_sampled_xan_seed_999_fr_en.csv
export TWEETSCOLUMN=${LANGUAGE}
export SYSTEMPROMT=${POLPOSTANNPATH}/prompts/system/system_prompt_${LANGUAGE}.txt
export USERPROMT=${POLPOSTANNPATH}/prompts/user/user_prompt_${TASK}_multiple_all_${LANGUAGE}.txt
export CHOICES="'Macron,Mélenchon,LePen,None'"

export OUTFOLDER=${POLPOSTANNPATH}/outputs_${SERVER}/v3ModelSelection${LANGUAGE}/${NAME}/guided/${TASK}/multiple/all

sbatch \
    --job-name=${NAME} \
    --output=${OUTFOLDER}/%j.log  \
    --error=${OUTFOLDER}/%j.out  \
    --ntasks-per-node=4 \
    --gres=gpu:h100:4 \
    --export=ALL \
    ${SERVER}/multipleChoicesAllPrompt_${GPU}_${SERVER}.slurm
