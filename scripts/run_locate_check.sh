#!/usr/bin/env bash
set -x
export VLLM_USE_V1=0
export TOKENIZERS_PARALLELISM=false

GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n1)

HYDRA_OVERRIDES=""

if echo "$GPU_NAME" | grep -q "V100"; then
    echo "[INFO] Detected V100 GPU, disabling unstable XLA/Torch optimizations..."
    export VLLM_ATTENTION_BACKEND=XFORMERS

    HYDRA_OVERRIDES="$HYDRA_OVERRIDES,llm.enable_chunked_prefill=false,llm.dtype=float32"
fi

if echo "$GPU_NAME" | grep -q "V100"; then
    CUDA_VISIBLE_DEVICES=2,3 python3 -m llm_evaluate.main_eval --config-name config_check_locate $HYDRA_OVERRIDES
else
    CUDA_VISIBLE_DEVICES=2,3 python3 -m llm_evaluate.main_eval --config-name config_check_locate 
fi
