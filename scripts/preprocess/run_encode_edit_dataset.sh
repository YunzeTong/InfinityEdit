#!/bin/bash
# ==============================================================================
# Pre-encode paired (source, edited) videos into VAE latents + T5 embeddings.
#
# One CSV per edit style, named "<style>.csv", with columns:
#   src_video_path, tgt_video_path, src_video_caption, edit_video_instruction
#
# Usage:
#   bash scripts/preprocess/run_encode_edit_dataset.sh --all
#   bash scripts/preprocess/run_encode_edit_dataset.sh simpsons_comic zoom_in
#   MAX_SAMPLES=50 bash scripts/preprocess/run_encode_edit_dataset.sh --all
# ==============================================================================
set -euo pipefail

#################################################################
## Config — EDIT THESE
#################################################################
SRC_ROOT="./data/edit_pairs"          # holds <style>.csv files
DST_ROOT="./data/train"               # output: <DST_ROOT>/<style>/*.pt
MODEL_PATH="./pretrained/Helios-Distilled"

NUM_GPUS=${NUM_GPUS:-4}
TARGET_HEIGHT=${TARGET_HEIGHT:-384}
TARGET_WIDTH=${TARGET_WIDTH:-640}
SRC_TARGET_FRAMES=${SRC_TARGET_FRAMES:-73}   # → 19 latent frames (history window)
TGT_TARGET_FRAMES=${TGT_TARGET_FRAMES:-33}   # →  9 latent frames (one chunk)
MAX_SAMPLES=${MAX_SAMPLES:-}                 # empty = encode all

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

#################################################################
## Which styles to process
#################################################################
STYLES=()
if [[ $# -eq 0 || "${1:-}" == "--all" ]]; then
    for csv in "${SRC_ROOT}"/*.csv; do
        STYLES+=("$(basename "${csv%.csv}")")
    done
    echo "=== Processing all ${#STYLES[@]} styles ==="
else
    STYLES=("$@")
    echo "=== Processing ${#STYLES[@]} styles ==="
fi

mkdir -p "$DST_ROOT"

for style in "${STYLES[@]}"; do
    CSV_PATH="${SRC_ROOT}/${style}.csv"
    if [[ ! -f "$CSV_PATH" ]]; then
        echo "[WARN] CSV not found, skipping: ${CSV_PATH}"
        continue
    fi

    OUTPUT_DIR="${DST_ROOT}/${style}"
    mkdir -p "$OUTPUT_DIR"

    echo ""
    echo "================================================================"
    echo "  Style:             ${style}"
    echo "  CSV:               ${CSV_PATH}"
    echo "  Output:            ${OUTPUT_DIR}"
    echo "  Src target frames: ${SRC_TARGET_FRAMES}"
    echo "  Tgt target frames: ${TGT_TARGET_FRAMES}"
    echo "  Max samples:       ${MAX_SAMPLES:-all}"
    echo "================================================================"

    MAX_SAMPLES_FLAG=""
    if [[ -n "$MAX_SAMPLES" ]]; then
        MAX_SAMPLES_FLAG="--max_samples $MAX_SAMPLES"
    fi

    COMMON_ARGS=(
        --csv "$CSV_PATH"
        --output_dir "$OUTPUT_DIR"
        --pretrained_model_name_or_path "$MODEL_PATH"
        --target_height "$TARGET_HEIGHT"
        --target_width "$TARGET_WIDTH"
        --src_target_frames "$SRC_TARGET_FRAMES"
        --tgt_target_frames "$TGT_TARGET_FRAMES"
    )

    if [[ $NUM_GPUS -gt 1 ]]; then
        accelerate launch \
            --num_processes="$NUM_GPUS" --num_machines=1 --mixed_precision=bf16 \
            "${PROJECT_ROOT}/scripts/preprocess/encode_edit_dataset.py" \
            "${COMMON_ARGS[@]}" $MAX_SAMPLES_FLAG
    else
        python "${PROJECT_ROOT}/scripts/preprocess/encode_edit_dataset.py" \
            "${COMMON_ARGS[@]}" $MAX_SAMPLES_FLAG
    fi

    echo "[DONE] ${style}"
done

echo ""
echo "=== All encoding tasks completed ==="
