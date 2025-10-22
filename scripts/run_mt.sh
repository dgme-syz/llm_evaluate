#!/usr/bin/env bash
set -e

export TOKENIZERS_PARALLELISM=false
export VLLM_WORKER_MULTIPROC_METHOD=spawn
# -------------------------------
# 参数解析
# -------------------------------
MODEL_PATH=""
LOG_PATH=""
EXTRA_ARGS=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --path)
            MODEL_PATH="$2"
            shift 2
            ;;
        --log)
            LOG_PATH="$2"
            shift 2
            ;;
        --extra)
            EXTRA_ARGS="$2"
            shift 2
            ;;
        *)
            shift
            ;;
    esac
done

# -------------------------------
# GPU 空闲检测函数
# -------------------------------
get_free_gpus() {
    mapfile -t free_gpus < <(
        nvidia-smi --query-gpu=index,memory.total,memory.used,utilization.gpu \
            --format=csv,noheader,nounits | while IFS=',' read -r idx total used util; do
                mem_free_percent=$(( 100 - 100*used/total ))
                util_int=${util// /}  # 去空格
                if (( mem_free_percent >= 95 && util_int < 10 )); then
                    echo $idx
                fi
            done
    )
    echo "${free_gpus[@]}"
}

# 获取空闲 GPU
while true; do
    AVAILABLE_GPUS=($(get_free_gpus))
    NUM_GPUS=${#AVAILABLE_GPUS[@]}

    if (( NUM_GPUS == 0 )); then
        echo "[INFO] No free GPUs detected. Waiting 10s before retry..."
        sleep 10
    else
        # 设置可用 GPU
        CUDA_VISIBLE_DEVICES=$(IFS=,; echo "${AVAILABLE_GPUS[*]}")
        echo "[INFO] Free GPUs detected: $CUDA_VISIBLE_DEVICES"

        # 根据可用 GPU 数量自动设置 tensor parallel size
        if (( NUM_GPUS >= 4 )); then
            TP_SIZE=4
        elif (( NUM_GPUS >= 2 )); then
            TP_SIZE=2
        else
            TP_SIZE=1
        fi

        echo "[INFO] Tensor parallel size set to $TP_SIZE"
        break  # 退出循环
    fi
done

echo "[INFO] Using GPUs: $CUDA_VISIBLE_DEVICES, tensor_parallel_size=$TP_SIZE"

# -------------------------------
# GPU 型号检测
# -------------------------------
GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n1)
echo "[INFO] Detected GPU: $GPU_NAME"

HYDRA_OVERRIDES=""

# 如果指定了模型路径，先添加到 HYDRA_OVERRIDES
if [ -n "$MODEL_PATH" ]; then
    HYDRA_OVERRIDES="llm.model=$MODEL_PATH llm.tokenizer=$MODEL_PATH"
fi

# 如果有额外参数，也加进去
if [ -n "$EXTRA_ARGS" ]; then
    if [ -n "$HYDRA_OVERRIDES" ]; then
        HYDRA_OVERRIDES="$HYDRA_OVERRIDES $EXTRA_ARGS"
    else
        HYDRA_OVERRIDES="$EXTRA_ARGS"
    fi
fi

# -------------------------------
# V100 特殊逻辑
# -------------------------------
export VLLM_USE_V1=0
if echo "$GPU_NAME" | grep -q "V100"; then
    echo "[INFO] Detected V100 GPU, disabling unstable XLA/Torch optimizations..."
    export VLLM_ATTENTION_BACKEND=XFORMERS

    if [ -n "$HYDRA_OVERRIDES" ]; then
        HYDRA_OVERRIDES="$HYDRA_OVERRIDES llm.enable_chunked_prefill=false llm.dtype=float32"
    else
        HYDRA_OVERRIDES="llm.enable_chunked_prefill=false llm.dtype=float32"
    fi
else
    echo "[INFO] Non-V100 GPU detected, running with default optimizations."
fi

# -------------------------------
# 自动设置 tensor_parallel_size
# -------------------------------
HYDRA_OVERRIDES="$HYDRA_OVERRIDES llm.tensor_parallel_size=$TP_SIZE"

# -------------------------------
# 执行 Python 命令
# -------------------------------
PY_CMD="VLLM_USE_V1=0 CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES python3 -m llm_evaluate.main_eval --config-name config_mt"

if [ -n "$HYDRA_OVERRIDES" ]; then
    PY_CMD="$PY_CMD $HYDRA_OVERRIDES"
fi

if [ -n "$LOG_PATH" ]; then
    echo "[INFO] Logging output to $LOG_PATH and printing to terminal"
    # 使用 tee 同时输出到终端和日志
    echo "$PY_CMD 2>&1 | tee $LOG_PATH"
    eval "$PY_CMD 2>&1 | tee $LOG_PATH"
else
    echo "$PY_CMD"
    eval "$PY_CMD"
fi