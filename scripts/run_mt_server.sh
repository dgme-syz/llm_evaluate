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
    HYDRA_OVERRIDES+=("llm.vllm.model=$MODEL_PATH" "llm.vllm.tokenizer=$MODEL_PATH")
fi
if [[ -n "$EXTRA_ARGS" ]]; then
    read -r -a EXTRA_ARRAY <<< "$EXTRA_ARGS"
    HYDRA_OVERRIDES+=("${EXTRA_ARRAY[@]}")
fi

PY_CMD=(python3 -m llm_evaluate.main_eval --config-name config_mt_server)
PY_CMD+=("${HYDRA_OVERRIDES[@]}")

[ -n "${CUDA_VISIBLE_DEVICES:-}" ] && export CUDA_VISIBLE_DEVICES

if [[ -n "$LOG_PATH" ]]; then
    echo "[INFO] Logging output to $LOG_PATH and printing to terminal"
    echo "${PY_CMD[@]}"
    "${PY_CMD[@]}" 2>&1 | tee "$LOG_PATH"
else
    echo "${PY_CMD[@]}"
    "${PY_CMD[@]}"
fi
