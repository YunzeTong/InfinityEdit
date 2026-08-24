#!/bin/bash
# ==============================================================================
# InfinityEdit — multi-round sequential editing inference.
#
# For every test sample, each edit round runs one adapter chunk followed by
# CHUNKS_PER_EDIT pyramid continuation chunks, and the history window carries
# over into the next round. Outputs per sample:
#   source.mp4, edit_{i}_{type}.mp4, full_video.mp4, generated_full.mp4, metadata.txt
#
#   bash scripts/run_sequential_edit.sh
# ==============================================================================
set -e

TIMESTAMP=$(date +"%m%d_%H%M%S")

################################################################################
# Paths — EDIT THESE
################################################################################
# Trained Edit Adapter checkpoint directory (a `checkpoint-XXXXX` folder).
ADAPTER_CKPT="./outputs/phase1/checkpoint-43000"

# Must be the same yaml used for training that checkpoint.
CONFIG="configs/edit_adapter_phase1.yaml"

# Frozen base video diffusion backbone.
BASE_MODEL_PATH="./pretrained/Helios-Distilled"

# Directory of source videos (*.mp4) or pre-encoded latents (*.pt).
DATA_DIR="./examples/videos"

# Benchmark CSV with edit instructions (200 samples, 3 edit rounds each).
TEST_CSV="./examples/benchmark.csv"

OUTPUT_DIR="./outputs/sequential_edit_${TIMESTAMP}"

# Set to the number of source files you want to process (2 shipped as examples).
NUM_SAMPLES=2

################################################################################
# Resources
################################################################################
NUM_GPUS=8
ACCELERATE_CONFIG="configs/accelerate_multi_gpu_${NUM_GPUS}card.yaml"

################################################################################
# Method knobs (paper settings)
################################################################################
USE_HISTORY_ADAPTER=true      # history cross-attention branch
USE_EMA=true                  # load the EMA weights from the checkpoint

EDIT_NUM_INFERENCE_STEPS=16   # adapter chunk: 16 (+1 terminal) steps
FINAL_SIGMA_EXTRA_STEP=0.01

PYRAMID_NUM_STAGES=3          # continuation chunks: 3-stage pyramid, 2 steps each
PYRAMID_STEPS="2 2 2"
CHUNKS_PER_EDIT=3             # continuation chunks per edit round (~6s of video);
                              # overridden per-row by the CSV's chunks_edit_N columns

SEED=42

################################################################################
# Assemble args
################################################################################
ARGS="--adapter_ckpt=${ADAPTER_CKPT} \
      --config=${CONFIG} \
      --data_dir=${DATA_DIR} \
      --output_dir=${OUTPUT_DIR} \
      --base_model_path=${BASE_MODEL_PATH} \
      --num_samples=${NUM_SAMPLES} \
      --test_csv=${TEST_CSV} \
      --edit_num_inference_steps=${EDIT_NUM_INFERENCE_STEPS} \
      --final_sigma_extra_step=${FINAL_SIGMA_EXTRA_STEP} \
      --pyramid_num_stages=${PYRAMID_NUM_STAGES} \
      --pyramid_steps ${PYRAMID_STEPS} \
      --chunks_per_edit=${CHUNKS_PER_EDIT} \
      --seed=${SEED}"

if [[ "${USE_EMA}" == true ]]; then
    ARGS="${ARGS} --use_ema"
fi
if [[ "${USE_HISTORY_ADAPTER}" == true ]]; then
    ARGS="${ARGS} --use_history_adapter"
fi

echo ""
echo "========================================="
echo "InfinityEdit — sequential editing inference"
echo "checkpoint:  ${ADAPTER_CKPT}"
echo "test csv:    ${TEST_CSV}"
echo "samples:     ${NUM_SAMPLES}"
echo "output:      ${OUTPUT_DIR}"
echo "edit:        ${EDIT_NUM_INFERENCE_STEPS}+1 steps, extra_step=${FINAL_SIGMA_EXTRA_STEP}"
echo "pyramid:     ${PYRAMID_NUM_STAGES} stages [${PYRAMID_STEPS}], ${CHUNKS_PER_EDIT} chunks/edit"
echo "gpus:        ${NUM_GPUS}"
echo "========================================="

accelerate launch --config_file "${ACCELERATE_CONFIG}" \
    sequential_edit/run_sequential_edit.py ${ARGS}
