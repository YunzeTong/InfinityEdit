#!/bin/bash
# ==============================================================================
# InfinityEdit — Stage 1 training (uniform sigma weighting, uniform frame loss)
#
# Trains the Edit Adapter on pre-encoded VAE latents. Runs on a single node with
# `accelerate launch`; set NUM_GPUS to the number of GPUs you have.
#
#   bash scripts/train_phase1.sh
# ==============================================================================
set -e

################################################################################
# Paths — EDIT THESE
################################################################################
# Root holding one subdirectory per edit style, each containing *.pt latents
# (train/<style>/*.pt). Point this at the downloaded latents, or at the output
# of the preprocessing step (see README, "Data preparation").
DATA_ROOT="./data/train"

# Base (frozen) video diffusion backbone. Must match model_config in the yaml.
CONFIG="configs/edit_adapter_phase1.yaml"

# Where checkpoints / logs go. Leave empty to use `output_dir` from the yaml.
RUN_DIR=""

################################################################################
# Resources
################################################################################
NUM_GPUS=8
ACCELERATE_CONFIG="configs/accelerate_multi_gpu_${NUM_GPUS}card.yaml"

################################################################################
# Edit styles to train on. "all" = every subdirectory of DATA_ROOT that has .pt
################################################################################
STYLES=(
    simpsons_comic
    ladudu_me
    move_up
    zoom_in
)

################################################################################
# Method knobs (these are the values used in the paper)
################################################################################
USE_MIXTURE_SIGMA=true      # mixture-of-Gaussians sigma sampling
FINAL_SIGMA_EXTRA_STEP=0.01 # extra terminal denoise step
WEIGHTING_SCHEME=""         # empty = use yaml's weighting_scheme

################################################################################
# Build the --data_dirs list
################################################################################
DATA_DIRS=()
if [[ ${#STYLES[@]} -eq 1 && "${STYLES[0]}" == "all" ]]; then
    for dir in "${DATA_ROOT}"/*/; do
        if ls "$dir"*.pt &>/dev/null; then
            DATA_DIRS+=("${DATA_ROOT}/$(basename "$dir")")
            echo "  + $(basename "$dir")"
        fi
    done
else
    for style in "${STYLES[@]}"; do
        if [[ -d "${DATA_ROOT}/${style}" ]]; then
            DATA_DIRS+=("${DATA_ROOT}/${style}")
            echo "  + ${style}"
        else
            echo "  [WARN] missing style directory, skipping: ${DATA_ROOT}/${style}"
        fi
    done
fi

if [[ ${#DATA_DIRS[@]} -eq 0 ]]; then
    echo "[ERROR] No valid style directories under ${DATA_ROOT}"
    exit 1
fi
echo "=== Training with ${#DATA_DIRS[@]} styles ==="

DATA_DIRS_STR="${DATA_DIRS[0]}"
for dir in "${DATA_DIRS[@]:1}"; do
    DATA_DIRS_STR="${DATA_DIRS_STR},${dir}"
done

################################################################################
# Assemble args
################################################################################
ARGS="--config=${CONFIG} \
      --data_dirs=${DATA_DIRS_STR} \
      --enable_temporal_self_attn"

# NOTE: no --reweighting_along_time in Stage 1 (uniform loss over frames).
if [[ -n "${RUN_DIR}" ]]; then
    ARGS="${ARGS} --output_dir=${RUN_DIR}"
fi
if [[ -n "${FINAL_SIGMA_EXTRA_STEP}" ]]; then
    ARGS="${ARGS} --final_sigma_extra_step=${FINAL_SIGMA_EXTRA_STEP}"
fi
if [[ -n "${WEIGHTING_SCHEME}" ]]; then
    ARGS="${ARGS} --weighting_scheme=${WEIGHTING_SCHEME}"
fi
if [[ "${USE_MIXTURE_SIGMA}" == true ]]; then
    ARGS="${ARGS} --use_mixture_sigma"
fi

echo ""
echo "========================================="
echo "InfinityEdit — Stage 1"
echo "config:      ${CONFIG}"
echo "gpus:        ${NUM_GPUS}"
echo "args:        ${ARGS}"
echo "========================================="

accelerate launch --config_file "${ACCELERATE_CONFIG}" \
    train_edit_adapter.py ${ARGS}
