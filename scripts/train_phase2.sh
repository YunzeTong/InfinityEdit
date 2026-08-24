#!/bin/bash
# ==============================================================================
# InfinityEdit — Stage 2 training (low-sigma focused schedule + temporal loss
# reweighting), resumed from a Stage 1 checkpoint.
#
# Set `resume_from_checkpoint` in configs/edit_adapter_phase2.yaml to the Stage 1
# checkpoint you want to continue from, then:
#
#   bash scripts/train_phase2.sh
# ==============================================================================
set -e

################################################################################
# Paths — EDIT THESE
################################################################################
DATA_ROOT="./data/train"   # train/<style>/*.pt
CONFIG="configs/edit_adapter_phase2.yaml"
RUN_DIR=""   # empty = use `output_dir` from the yaml

################################################################################
# Resources
################################################################################
NUM_GPUS=8
ACCELERATE_CONFIG="configs/accelerate_multi_gpu_${NUM_GPUS}card.yaml"

STYLES=(
    simpsons_comic
    ladudu_me
    move_up
    zoom_in
)

USE_MIXTURE_SIGMA=true
FINAL_SIGMA_EXTRA_STEP=0.01
WEIGHTING_SCHEME=""

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
# Assemble args — Stage 2 adds --reweighting_along_time
################################################################################
ARGS="--config=${CONFIG} \
      --data_dirs=${DATA_DIRS_STR} \
      --enable_temporal_self_attn \
      --reweighting_along_time"

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
echo "InfinityEdit — Stage 2"
echo "config:      ${CONFIG}"
echo "gpus:        ${NUM_GPUS}"
echo "args:        ${ARGS}"
echo "========================================="

accelerate launch --config_file "${ACCELERATE_CONFIG}" \
    train_edit_adapter.py ${ARGS}
