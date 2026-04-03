#!/bin/bash
# TurboQuant v2 - Cloud run script for 8×H100
# Usage: bash run_cloud.sh [SEED]

set -euo pipefail

SEED=${1:-1337}

echo "=== TurboQuant v2 Cloud Run ==="
echo "Seed: $SEED"
echo "Config: 11L/512/3x + PolarQuant + QJL + Int6 QAT + EMA"

export RUN_ID="turboquant_v2_seed${SEED}"
export SEED="$SEED"

# Model
export NUM_LAYERS=11
export MODEL_DIM=512
export MLP_MULT=3
export NUM_HEADS=8
export NUM_KV_HEADS=4

# Training (tuned)
export MATRIX_LR=0.025
export SCALAR_LR=0.025
export TIED_EMBED_LR=0.035
export MUON_MOMENTUM=0.99
export MUON_MOMENTUM_WARMUP_START=0.92
export MUON_MOMENTUM_WARMUP_STEPS=1500
export MUON_WEIGHT_DECAY=0.04
export ADAM_WEIGHT_DECAY=0.04
export GRAD_CLIP_NORM=0.3
export WARMDOWN_ITERS=3500

# Compression
export QAT_ENABLED=1
export QAT_BITS=6
export QAT_LATE_THRESHOLD=0.15
export EMA_ENABLED=1
export EMA_DECAY=0.997

# Sliding window eval
export SLIDING_EVAL=1
export SLIDING_EVAL_STRIDE=64

torchrun --standalone --nproc_per_node=8 train_gpt.py
