LANGUAGE=$1
TASK=$2
SEED=$3

export MODELPARAMS="'{\"model\": \"meta-llama/Llama-3.3-70B-Instruct\", \"guided_decoding_backend\": \"xgrammar\", \"max_model_len\": 1200, \"seed\": ${SEED}, \"tensor_parallel_size\": 2}'"
export SAMPLINGPARAMS="'{\"seed\": ${SEED}, \"max_tokens\": 256}'"
export NAME=Llama-3.3-70B-Instruct-seed${SEED}

export TWEETSFILE=${POLPOSTANNPATH}/400_balanced_sampled_xan_seed_999_fr_en.csv
export TWEETSCOLUMN=${LANGUAGE}
export SYSTEMPROMT=${POLPOSTANNPATH}/prompts/system/system_prompt_${LANGUAGE}.txt
export USERPROMT=${POLPOSTANNPATH}/prompts/user/user_prompt_${TASK}_multiple_all_${LANGUAGE}.txt

export OUTFOLDER=${POLPOSTANNPATH}/outputs/v3ModelSelection${LANGUAGE}/${NAME}/guided/${TASK}/multiple/all

sbatch \
    --job-name=${NAME} \
    --output=${OUTFOLDER}/%j.log  \
    --error=${OUTFOLDER}/%j.out  \
    --ntasks-per-node=2 \
    --gres=gpu:h100:2 \
    --export=ALL \
    multipleChoicesAllPrompt_h100_jeanzay.slurm
