#!/usr/bin/env bash
set -euo pipefail

# ----------------------------------------
# Script: Run multiple models
# Notes:
# - No data_args needed; datasets handled by Python Hydra config
# - Adds retry logic
# ----------------------------------------

declare -A models
# models["Seed-X-PPO-7B"]="/home/nfs05/shenyz/models/Seed-X-PPO-7B sample_params.offline.temperature=0.0"
models["Hunyuan-MT-7B"]="/home/nfs05/shenyz/models/Hunyuan-MT-7B sample_params.offline.temperature=0.7 sample_params.offline.top_p=0.6 sample_params.offline.top_k=20 sample_params.offline.repetition_penalty=1.05"
# models["Qwen3-8B"]="/home/nfs06/shenyz/models/Qwen3-8B sample_params.offline.temperature=0.6 sample_params.offline.top_p=0.95 sample_params.offline.top_k=20"
# models["Qwen2.5-7B-Instruct"]="/home/nfs06/shenyz/models/Qwen2.5-7B-Instruct sample_params.offline.repetition_penalty=1.05 sample_params.offline.temperature=0.7 sample_params.offline.top_p=0.8 sample_params.offline.top_k=20"
# models["Qwen3-0.6B"]="/home/nfs06/shenyz/models/Qwen3-0.6B sample_params.offline.temperature=0.6 sample_params.offline.top_p=0.95 sample_params.offline.top_k=20"
# models["Qwen3-4B"]="/home/nfs06/shenyz/models/Qwen3-4B sample_params.offline.temperature=0.6 sample_params.offline.top_p=0.95 sample_params.offline.top_k=20"
# models["Qwen3-4B-Thinking-2507"]="/home/nfs05/shenyz/models/Qwen3-4B-Thinking-2507 sample_params.offline.temperature=0.6 sample_params.offline.top_p=0.95 sample_params.offline.top_k=20 sample_params.offline.max_tokens=8192"

LOG_DIR="./logs"
mkdir -p "$LOG_DIR"

MAX_RETRIES=3
task_idx=0
total_models=${#models[@]}
start_time_all=$(date +%s)

for model_name in "${!models[@]}"; do
    task_idx=$((task_idx + 1))
    model_path=$(echo "${models[$model_name]}" | awk '{print $1}')
    model_extra=$(echo "${models[$model_name]}" | cut -d' ' -f2-)
    log_name="${LOG_DIR}/mt_${model_name}.log"

    echo "──────────────────────────────────────────────"
    echo "▶ [${task_idx}/${total_models}] Running $model_name"
    echo "   Log: $log_name"
    start_time=$(date +%s)

    # Retry mechanism
    attempt=0
    while (( attempt < MAX_RETRIES )); do
        attempt=$(( attempt + 1 ))
        echo "⏳ Attempt #$attempt..."

        bash scripts/run_mt.sh \
            --path "$model_path" \
            --log "$log_name" \
            --extra "$model_extra"

        exit_code=$?
        if [[ $exit_code -eq 0 ]]; then
            echo "✅ Success on attempt #$attempt"
            break
        else
            echo "⚠️ Task failed with exit code $exit_code."
            if (( attempt < MAX_RETRIES )); then
                echo "🔁 Retrying..."
            else
                echo "❌ Max retries reached. Skipping..."
            fi
        fi
    done

    end_time=$(date +%s)
    duration=$(( end_time - start_time ))
    echo "✅ Finished: $model_name — took ${duration}s"
    echo
done

end_time_all=$(date +%s)
total_duration=$(( end_time_all - start_time_all ))
echo "🏁 All tasks completed in ${total_duration}s"
echo "──────────────────────────────────────────────"
