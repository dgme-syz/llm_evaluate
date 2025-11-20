#!/usr/bin/env bash
set -euo pipefail

MODEL_PATH=""
LOG_PATH=""
EXTRA_ARGS=""
export TRANSFORMERS_OFFLINE=1

while [[ $# -gt 0 ]]; do
    case $1 in
        --path)
            if [[ -z "${2:-}" ]]; then
                echo "[ERROR] --path requires an argument"
                exit 1
            fi
            MODEL_PATH="$2"
            shift 2
            ;;
        --log)
            if [[ -z "${2:-}" ]]; then
                echo "[ERROR] --log requires an argument"
                exit 1
            fi
            LOG_PATH="$2"
            shift 2
            ;;
        --extra)
            if [[ -z "${2:-}" ]]; then
                echo "[ERROR] --extra requires an argument"
                exit 1
            fi
            EXTRA_ARGS="$2"
            shift 2
            ;;
        *)
            echo "[WARN] Unknown argument: $1"
            shift
            ;;
    esac
done

HYDRA_OVERRIDES=()
if [[ -n "$MODEL_PATH" ]]; then
    HYDRA_OVERRIDES+=("server.model=$MODEL_PATH")
fi
if [[ -n "$EXTRA_ARGS" ]]; then
    read -r -a EXTRA_ARRAY <<< "$EXTRA_ARGS"
    HYDRA_OVERRIDES+=("${EXTRA_ARRAY[@]}")
fi

# ------------------------
# 使用 get_free_gpus 函数选择未占用的 CUDA
# ------------------------
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

AVAILABLE_GPUS=$(get_free_gpus)
if [[ -z "$AVAILABLE_GPUS" ]]; then
    echo "[WARN] No free GPU found, using default CUDA_VISIBLE_DEVICES"
else
    export CUDA_VISIBLE_DEVICES=$(echo "$AVAILABLE_GPUS" | tr ' ' ',')
    echo "[INFO] Using available GPU(s): $CUDA_VISIBLE_DEVICES"
fi

PY_CMD=(python3 -m llm_evaluate.main_eval --config-name config_mt_server)
PY_CMD+=("${HYDRA_OVERRIDES[@]}")

if [[ -n "$LOG_PATH" ]]; then
    echo "[INFO] Logging output to $LOG_PATH and printing to terminal"
    echo "${PY_CMD[@]}"
    "${PY_CMD[@]}" 2>&1 | tee "$LOG_PATH"
else
    echo "${PY_CMD[@]}"
    "${PY_CMD[@]}"
fi
