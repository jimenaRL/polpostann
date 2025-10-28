LANGUAGE=$1
TASK=$2
SEED=$3
GPU=$4

export MODELPARAMS="'{\"model\": \"mistralai/Mistral-Small-24B-Instruct-2501\", \"tokenizer_mode\": \"mistral\", \"config_format\": \"mistral\", \"load_format\": \"mistral\", \"guided_decoding_backend\": \"xgrammar\", \"seed\": ${SEED}, \"tensor_parallel_size\": 1}'"
export SAMPLINGPARAMS="'{\"temperature\": 0.15, \"seed\": ${SEED}, \"max_tokens\": 256}'"
export NAME=Mistral-Small-24B-Instruct-2501-seed${SEED}

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
    --ntasks-per-node=1 \
    --gres=gpu:h100:1 \
    --export=ALL \
    ${SERVER}/annotate_tweets_${GPU}_${SERVER}.slurm
