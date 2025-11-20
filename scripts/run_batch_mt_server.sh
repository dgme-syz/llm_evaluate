#!/usr/bin/env bash
set -euo pipefail

if [[ -z "$DEEPSEEK_API_KEY" ]]; then
    echo "DASHSCOPE_API_KEY is not set!"
    exit 1
fi

declare -A models
models["deepseek-reasoner"]="'deepseek-reasoner' server.openai_args.api_key='${DEEPSEEK_API_KEY}' server.openai_args.base_url='https://api.deepseek.com/v1' ++llm.prompt_template='null'"


LOG_DIR="./logs"
mkdir -p "$LOG_DIR"

for model_name in "${!models[@]}"; do
    model_path=$(echo "${models[$model_name]}" | awk '{print $1}')
    model_extra=$(echo "${models[$model_name]}" | cut -d' ' -f2-)
    log_name="${LOG_DIR}/mt_${model_name}.log"

    start_time=$(date +%s)

    echo "Starting processing model: $model_name"
    echo "Log file: $log_name"
    echo "Command: bash scripts/run_mt_server.sh --path $model_path --log $log_name --extra $model_extra"

    bash scripts/run_mt_server.sh --path "$model_path" --log "$log_name" --extra "$model_extra" 

    end_time=$(date +%s)
    duration=$((end_time - start_time))

    echo -e "\nModel $model_name completed in $duration seconds."

done