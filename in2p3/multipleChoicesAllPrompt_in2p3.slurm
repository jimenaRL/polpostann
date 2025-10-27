#!/bin/bash

#SBATCH --partition=gpu               # set gpu partition

# We use a single node with several gpus and let vllm manage gpus paralellisation
# with the 'tensor_parallel_size' model parameter.
# The number of gpus used is defined by the paramametes 'gres' and 'ntasks-per-node'
# which are given as input variable when launching this slurm script.
# We do this to used the same script with different gpus configs for each llm model.

#SBATCH --ntasks=1                    # Exécuter une seule tâche
#SBATCH --mem=128G                    # Mémoire en MB par défaut
#SBATCH --time=6-23:59                # Max temps execution en format « jours-heures:minutes »

#SBATCH --mail-user=jimena.royoletelier@sciencespo.fr # Où envoyer l'e-mail
#SBATCH --mail-type=ALL                # Événements déclencheurs (NONE, BEGIN, END, FAIL, ALL)

# Cleans out modules loaded in interactive and inherited by default
module purge

# Load python environement
module load Programming_Languages/python/3.11.4

# Gives access to the modules compatible with the gpu_p6 partition
source /sps/humanum/user/jroyolet/environments/vllm-0.10.1/bin/activate

# Show shell used
echo "SHELL:"
echo $0

# show python used
echo "PYTHON:"
echo $(python --version)
echo $(which python)

# show gpus availables
nvidia-smi

# Code execution
export SCRIPT=/sps/humanum/user/jroyolet/dev/polpostann/annotate_tweets.py

echo ""
echo "SCRIPT:"
echo "${SCRIPT}"

echo ""
echo "MODELPARAMS:"
echo "${MODELPARAMS}"

echo ""
echo "SAMPLINGPARAMS:"
echo "${SAMPLINGPARAMS}"

echo ""
echo "OUTFOLDER:"
echo "${OUTFOLDER}"


cmd="python ${SCRIPT} \
       --model_params=${MODELPARAMS} \
       --sampling_params=${SAMPLINGPARAMS} \
       --tweets_file=${TWEETSFILE} \
       --tweets_column=${TWEETSCOLUMN} \
       --system_prompt=${SYTEMPROMT} \
       --user_prompt=${USERPROMT} \
       --guided_choice=${CHOICES} \
       --outfolder=${OUTFOLDER}"
echo "[RUNNING] ${cmd}"

eval "$cmd"
