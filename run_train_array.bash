#!/usr/bin/env bash
#SBATCH -J tabedt
#SBATCH -N 1 --gpus-per-node=1
#SBATCH -c 8
#SBATCH -t 0-96:00:00
#SBATCH --array=0-9
#SBATCH -o TabEDT_%x_seed%a_%A.out
#
# Meta-training with the thesis settings: 10 seeds (0-9) per depth, brier reward
# (the size-weighted Gini reward of the thesis), candidate cap M = 128 in
# meta-training, 40,000 iterations, CC-18 evaluation every 100 iterations and a
# checkpoint every 500. Submit once per depth:
#
#   sbatch --export=ALL,DEPTH=3,MOTHERNET_CKPT=/path/to/mothernet.cpkt run_train_array.bash
#
# Resubmitting the same line resumes every seed from its last checkpoint.
# Load your environment first, for example:
#   module load Python/3.x ; source /path/to/venv/bin/activate ; cd /path/to/Amortized_Reinforcement_Learning_with_Tabular_Foundation_Models

: "${WANDB_API_KEY:?set your W&B API key in the environment}"
DEPTH=${DEPTH:?set DEPTH to 3, 4 or 5}
MOTHERNET_CKPT=${MOTHERNET_CKPT:?set MOTHERNET_CKPT to the MotherNet checkpoint}
PRIOR_CONFIG=${PRIOR_CONFIG:-prior_config_1.pkl}
ITERS=${ITERS:-40000}
TAG=${TAG:-d${DEPTH}_rerun}       # W&B run id synth_brier_seed{k}_{TAG}; avoids the thesis runs
SEED=$SLURM_ARRAY_TASK_ID

export TABEDT_MAX_DEPTH=$DEPTH

python meta_tabedt_synthetic.py \
    --reward brier \
    --max-candidates 128 \
    --iterations "$ITERS" \
    --eval-every 100 \
    --checkpoint-every 500 \
    --seed "$SEED" \
    --tag "$TAG" \
    --checkpoint "$MOTHERNET_CKPT" \
    --prior-config "$PRIOR_CONFIG" \
    --prior-device cuda
