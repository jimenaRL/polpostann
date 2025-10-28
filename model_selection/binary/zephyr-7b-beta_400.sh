LANGUAGE=$1    # frech
TASK=$2        # criticism
SEED=$3        # 1
GPU=$4         # h100
CANDIDATE=$5   # macron

export MODELPARAMS="'{\"model\": \"HuggingFaceH4/zephyr-7b-beta\", \"guided_decoding_backend\": \"xgrammar\", \"seed\": ${SEED}, \"gpu_memory_utilization\": 0.9}'"
export SAMPLINGPARAMS="'{\"temperature\": 0.7, \"top_p\": 0.95, \"top_k\": 50, \"max_tokens\": 16, \"repetition_penalty\": 1.2, \"seed\": ${SEED}}'"
export NAME=zephyr-7b-beta-seed${SEED}

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
