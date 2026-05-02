#!/usr/bin/env bash
set -euo pipefail

declare -A models
# models["deepseek-reasoner"]="'deepseek-reasoner' server.openai_args.api_key='${DEEPSEEK_API_KEY}' server.openai_args.base_url='https://api.deepseek.com/v1' ++server.prompt_template=qwen_chat_mt"
# models["DeepSeek-V3.2-Exp"]="'DeepSeek-V3.2-Exp' server.openai_args.api_key=sk-REDACTED server.openai_args.base_url='https://llmapi.paratera.com/v1' ++server.prompt_template=qwen_chat_mt"
# models["DeepSeek-V3.1"]="'DeepSeek-V3.1' server.openai_args.api_key=sk-REDACTED server.openai_args.base_url='https://llmapi.paratera.com/v1' ++server.prompt_template=qwen_chat_mt"
# models["gemini-2.0-flash"]="'gemini-2.0-flash' server.openai_args.api_key=sk-REDACTED server.openai_args.base_url='https://api.bltcy.ai/v1/' ++server.prompt_template=qwen_chat_mt"
# models["gemini-1.5-flash"]="'gemini-1.5-flash' server.openai_args.api_key=sk-REDACTED server.openai_args.base_url='https://api.bltcy.ai/v1/' ++server.prompt_template=qwen_chat_mt"
# models["qwen-max"]="'qwen-max' server.openai_args.api_key=sk-REDACTED server.openai_args.base_url='https://api.bltcy.ai/v1/' ++server.prompt_template=qwen_chat_mt"
models["deepseek-v3.2"]="'deepseek-v3.2' server.openai_args.api_key=sk-REDACTED server.openai_args.base_url='https://api.bltcy.ai/v1/' ++server.prompt_template=qwen_chat_mt"
# models["DeepSeek-V3"]="'deepseek-v3' server.openai_args.api_key=sk-REDACTED server.openai_args.base_url='https://api.bltcy.ai/v1/' ++server.prompt_template=qwen_chat_mt"
# models["gpt-5.2"]="'gpt-5.2' server.openai_args.api_key=sk-REDACTED server.openai_args.base_url='https://api.bltcy.ai/v1/' ++server.prompt_template=qwen_chat_mt"

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
