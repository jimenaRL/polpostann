PARTITION=$1
NBGPUS=$2

export MODELPARAMS="'{\"model\": \"mistralai/Mistral-Small-24B-Instruct-2501\", \"tokenizer_mode\": \"mistral\", \"config_format\": \"mistral\", \"load_format\": \"mistral\", \"guided_decoding_backend\": \"xgrammar\", \"seed\": 1, \"tensor_parallel_size\": ${NBGPUS}}'"
export SAMPLINGPARAMS="'{\"temperature\": 0.15, \"seed\": 1, \"max_tokens\": 256}'"
export NAME=Mistral-Small-24B-Instruct-2501-seed1

export TWEETSFILE=/lustre/fswork/projects/rech/nmf/umu89ib/dev/polpostann/enumerated_cleaned_text2annotate_2022-03-27_2022-04-25.csv
export TWEETSCOLUMN=french
export SYSTEMPROMT=/lustre/fswork/projects/rech/nmf/umu89ib/dev/polpostann/prompts/system/system_prompt_french.txt
export USERPROMT=/lustre/fswork/projects/rech/nmf/umu89ib/dev/polpostann/prompts/user/user_prompt_voteintention_multiple_all_french.txt

export BASEPATH=/lustre/fswork/projects/rech/nmf/umu89ib/dev/polpostann
export OUTFOLDER=${BASEPATH}/outputs/french/${NAME}/guided/voteintention/multiple/all

echo ${PARTITION}
echo ${NBGPUS}

sbatch \
    --job-name=${NAME} \
    --ntasks-per-node=${NBGPUS} \
    --gres=gpu:${PARTITION}:${NBGPUS} \
    --output=${OUTFOLDER}/%j.log  \
    --error=${OUTFOLDER}/%j.out  \
    multipleChoicesAllPrompt_${PARTITION}_${SERVER}.slurm
