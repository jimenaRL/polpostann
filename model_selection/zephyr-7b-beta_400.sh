LANGUAGE=$1
TASK=$2
SEED=$3

export MODELPARAMS="'{\"model\": \"HuggingFaceH4/zephyr-7b-beta\", \"guided_decoding_backend\": \"xgrammar\", \"seed\": ${SEED}, \"gpu_memory_utilization\": 0.9}'"
export SAMPLINGPARAMS="'{\"temperature\": 0.7, \"top_p\": 0.95, \"top_k\": 50, \"max_tokens\": 16, \"repetition_penalty\": 1.2, \"seed\": ${SEED}}'"
export NAME=zephyr-7b-beta-seed${SEED}

export TWEETSFILE=${POLPOSTANNPATH}/400_balanced_sampled_xan_seed_999_fr_en.csv
export TWEETSCOLUMN=${LANGUAGE}
export SYSTEMPROMT=${POLPOSTANNPATH}/prompts/system/system_prompt_${LANGUAGE}.txt
export USERPROMT=${POLPOSTANNPATH}/prompts/user/user_prompt_${TASK}_multiple_all_${LANGUAGE}.txt

export OUTFOLDER=${POLPOSTANNPATH}/outputs/v3ModelSelection${LANGUAGE}/${NAME}/guided/${TASK}/multiple/all

sbatch \
    --job-name=${NAME} \
    --output=${OUTFOLDER}/%j.log  \
    --error=${OUTFOLDER}/%j.out  \
    --ntasks-per-node=1 \
    --gres=gpu:h100:1 \
    --export=ALL \
    multipleChoicesAllPrompt_h100_jeanzay.slurm
