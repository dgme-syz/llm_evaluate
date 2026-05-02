#!/usr/bin/env bash
set -euo pipefail

# ----------------------------------------
# Script: Run multiple models
# Notes:
# - No data_args needed; datasets handled by Python Hydra config
# - Adds retry logic
# ----------------------------------------

THINKING=""
export NUMEXPR_MAX_THREADS=64

declare -A models
# models["Qwen2.5-7B"]="/home/nfs05/model/Qwen2.5-7B generate_params.offline.sampling_params.temperature=0.6 ++generate_params.offline.sampling_params.top_p=0.95 ++generate_params.offline.sampling_params.top_k=20 ++llm.prompt_template=qwen_chat_mt ++tokenize_args.enable_thinking=false ++generate_params.offline.sampling_params.repetition_penalty=1.05 ++llm.prompt_template=qwen_chat_mt ++tokenize_args.enable_thinking=false ++evaluate.eval_func=base ++evaluate.metrics_args.ignore_columns=[] ++llm.exp_tag=Qwen2.5-7B-Base"
models["Qwen2.5-7B-Instruct"]="/home/nfs05/model/Qwen2.5-7B-Instruct generate_params.offline.sampling_params.temperature=0.6 ++generate_params.offline.sampling_params.top_p=0.95 ++generate_params.offline.sampling_params.top_k=20 ++llm.prompt_template=qwen_chat_mt ++tokenize_args.enable_thinking=false ++generate_params.offline.sampling_params.repetition_penalty=1.05 ++llm.prompt_template=qwen_chat_mt ++tokenize_args.enable_thinking=false ++evaluate.eval_func=base ++evaluate.metrics_args.ignore_columns=[] ++llm.exp_tag=Qwen2.5-7B-Instruct"
# models["ours-4b_en2fi"]="/home/nfs05/shenyz/translation/verl/bs@128_n@8_m@1_@20251224_212804_@Qwen3-4B-mix_en_fi_mt-noqe_@4gpus/global_step_105/actor/huggingface generate_params.offline.sampling_params.temperature=0.6 ++generate_params.offline.sampling_params.top_p=0.95 ++generate_params.offline.sampling_params.top_k=20 ++generate_params.offline.sampling_params.repetition_penalty=1.05 ++llm.prompt_template=qwen_chat_mt ++llm.exp_tag=ours-4b_en2fi"
# models["mt-r1-zero_en2fi"]="/home/nfs05/shenyz/translation/verl/bs@128_n@72_m@1_@20251223_213631_@Qwen3-4B-mix_en_fi_mt-zero-72_@4gpus/global_step_105/actor/huggingface generate_params.offline.sampling_params.temperature=0.6 ++generate_params.offline.sampling_params.top_p=0.95 ++generate_params.offline.sampling_params.top_k=20 ++generate_params.offline.sampling_params.repetition_penalty=1.05 ++llm.prompt_template=qwen_chat_mt ++evaluate.eval_func=base ++evaluate.metrics_args.ignore_columns=[] ++llm.exp_tag=mt-r1-zero_en2fi"
# models["Seed-X-PPO-7B"]="/home/nfs05/shenyz/models/Seed-X-PPO-7B generate_params.offline.sampling_params.temperature=0.0 ++llm.prompt_template=seed_x_mt ++evaluate.eval_func=base ++evaluate.metrics_args.ignore_columns=[] ++llm.exp_tag=seed_x_ppo_7B"
# models["Tower-Plus-9B"]="/home/nfs05/model/Tower-Plus-9B generate_params.offline.sampling_params.temperature=0.0 ++llm.prompt_template=tower_plus_input ++evaluate.eval_func=base ++evaluate.metrics_args.ignore_columns=[] ++llm.exp_tag=Tower-Plus-9B"
# models["TowerInstruct-13B-v0.1"]="/home/nfs05/model/TowerInstruct-13B-v0.1/ generate_params.offline.sampling_params.temperature=0.0 ++llm.prompt_template=tower_chat_input ++evaluate.eval_func=base ++evaluate.metrics_args.ignore_columns=[] ++llm.exp_tag=tower-13B-v0.1"
# models["Hunyuan-MT-7B"]="/home/nfs05/shenyz/models/Hunyuan-MT-7B generate_params.offline.sampling_params.temperature=0.7 ++generate_params.offline.sampling_params.top_p=0.6 ++generate_params.offline.sampling_params.top_k=20 ++generate_params.offline.sampling_params.repetition_penalty=1.05 ++llm.prompt_template=hunyuan_mt_chat ++evaluate.eval_func=base ++llm.prompt_template=hunyuan_mt ++evaluate.metrics_args.ignore_columns=[]"
# models["SSR-X-Zero-7B"]="/home/nfs06/shenyz/models/SSR-X-Zero-7B/ ++generate_params.offline.sampling_params.temperature=0 ++llm.exp_tag=SSR-X-Zero-7B ++llm.prompt_template=ssr_x_mt ++evaluate.metrics_args.ignore_columns=[] ++evaluate.eval_func=base"
# models["Qwen2.5-7B-Instruct"]="/home/nfs06/shenyz/models/Qwen2.5-7B-Instruct generate_params.offline.sampling_params.repetition_penalty=1.05 generate_params.offline.sampling_params.temperature=0.7 generate_params.offline.sampling_params.top_p=0.8 generate_params.offline.sampling_params.top_k=20 ++llm.prompt_template=qwen_chat_mt ${THINKING}"
# models["Qwen3-1.7B"]="/home/nfs06/shenyz/models/Qwen3-1.7B generate_params.offline.sampling_params.temperature=0.6 ++generate_params.offline.sampling_params.top_p=0.95 ++generate_params.offline.sampling_params.top_k=20 ++llm.prompt_template=qwen_chat_mt ++llm.exp_tag=Qwen3-1.7B-no-think ${THINKING} ++tokenize_args.enable_thinking=false ++evaluate.eval_func_args.check_num=0 ++evaluate.metrics_args.ignore_columns=[]"
# models["Qwen3-4B"]="/home/nfs06/shenyz/models/Qwen3-4B generate_params.offline.sampling_params.temperature=0.6 ++generate_params.offline.sampling_params.top_p=0.95 ++generate_params.offline.sampling_params.top_k=20 ++llm.prompt_template=qwen_chat_mt ++llm.exp_tag=Qwen3-4B-no-think ++tokenize_args.enable_thinking=false ++evaluate.eval_func_args.check_num=0 ++evaluate.metrics_args.ignore_columns=[]"
# models["Qwen3-8B"]="/home/nfs06/shenyz/models/Qwen3-8B generate_params.offline.sampling_params.temperature=0.6 ++generate_params.offline.sampling_params.top_p=0.95 ++generate_params.offline.sampling_params.top_k=20 ++generate_params.offline.sampling_params.repetition_penalty=1.05 ++llm.prompt_template=qwen_chat_mt ++tokenize_args.enable_thinking=false ++evaluate.eval_func=base ++evaluate.metrics_args.ignore_columns=[] ++llm.exp_tag=Qwen3-8B"
# models["Qwen3-14B"]="/home/nfs05/model/Qwen3-14B generate_params.offline.sampling_params.temperature=0.6 ++generate_params.offline.sampling_params.top_p=0.95 ++generate_params.offline.sampling_params.top_k=20 ++generate_params.offline.sampling_params.repetition_penalty=1.05 ++llm.prompt_template=qwen_chat_mt ++tokenize_args.enable_thinking=false ++evaluate.eval_func=base ++evaluate.metrics_args.ignore_columns=[] ++llm.exp_tag=Qwen3-14B"
# models["Qwen3-32B"]="/home/nfs05/model/Qwen3-32B generate_params.offline.sampling_params.temperature=0.6 ++generate_params.offline.sampling_params.top_p=0.95 ++generate_params.offline.sampling_params.top_k=20 ++generate_params.offline.sampling_params.repetition_penalty=1.05 ++llm.prompt_template=qwen_chat_mt ++tokenize_args.enable_thinking=false ++evaluate.eval_func=base ++evaluate.metrics_args.ignore_columns=[] ++llm.exp_tag=Qwen3-32B"
# models["Qwen3-4B-Thinking-2507"]="/home/nfs05/model/Qwen3-4B-Thinking-2507 generate_params.offline.sampling_params.temperature=0.6 ++generate_params.offline.sampling_params.top_p=0.95 ++generate_params.offline.sampling_params.top_k=20 ++generate_params.offline.sampling_params.repetition_penalty=1.05 ++llm.prompt_template=qwen_chat_mt ++tokenize_args.enable_thinking=false ++evaluate.eval_func=base ++evaluate.metrics_args.ignore_columns=[] ++llm.exp_tag=Qwen3-4B-Thinking-2507"
# models["Qwen3-4B-Instruct-2507"]="/home/nfs05/model/Qwen3-4B-Instruct-2507 generate_params.offline.sampling_params.temperature=0.6 ++generate_params.offline.sampling_params.top_p=0.95 ++generate_params.offline.sampling_params.top_k=20 ++generate_params.offline.sampling_params.repetition_penalty=1.05 ++llm.prompt_template=qwen_chat_mt ++tokenize_args.enable_thinking=false ++evaluate.eval_func=base ++evaluate.metrics_args.ignore_columns=[] ++llm.exp_tag=Qwen3-4B-Instruct-2507"
# models["Qwen3-30B-A3B"]="/home/nfs05/model/Qwen3-30B-A3B generate_params.offline.sampling_params.temperature=0.6 ++generate_params.offline.sampling_params.top_p=0.95 ++generate_params.offline.sampling_params.top_k=20 ++generate_params.offline.sampling_params.repetition_penalty=1.05 ++llm.prompt_template=qwen_chat_mt ++tokenize_args.enable_thinking=false ++evaluate.eval_func=base ++evaluate.metrics_args.ignore_columns=[] ++llm.exp_tag=Qwen3-30B-A3B"
# models["Qwen3-0.6B"]="/home/nfs06/shenyz/models/Qwen3-0.6B generate_params.offline.sampling_params.temperature=0.6 ++generate_params.offline.sampling_params.top_p=0.95 ++generate_params.offline.sampling_params.top_k=20 ++llm.prompt_template=qwen_chat_mt ++llm.exp_tag=Qwen3-0.6B-no-think ++tokenize_args.enable_thinking=false ++evaluate.eval_func_args.check_num=0 ++evaluate.metrics_args.ignore_columns=[]"
# models["Qwen3-4B-trained"]="/home/nfs05/shenyz/translation/verl/bs@128_n@8_m@2_@20251117_234641/global_step_230/actor/huggingface/ generate_params.offline.sampling_params.temperature=0.6 ++generate_params.offline.sampling_params.top_p=0.95 ++generate_params.offline.sampling_params.top_k=20 ++generate_params.offline.sampling_params.repetition_penalty=1.15 ++llm.prompt_template=qwen_chat_mt ++llm.exp_tag=Qwen3-4B-trained ${THINKING}"
# models["Qwen3-0.6B-trained"]="/home/nfs05/shenyz/translation/verl/bs@128_n@8_@20251108_115419/global_step_230/actor/huggingface/ generate_params.offline.sampling_params.temperature=0.6 ++generate_params.offline.sampling_params.top_p=0.95 ++generate_params.offline.sampling_params.top_k=20 ++generate_params.offline.sampling_params.repetition_penalty=1.05 ++llm.prompt_template=qwen_chat_mt ++llm.exp_tag=Qwen3-0.6B-trained ${THINKING}"
# models["Qwen3-4B-mt-zero"]="/home/nfs05/shenyz/translation/verl/bs@128_n@8_m@1_@20251122_052744_@Qwen3-4B-mt-zero/global_step_105/actor/huggingface/ generate_params.offline.sampling_params.temperature=0.6 ++generate_params.offline.sampling_params.top_p=0.95 ++generate_params.offline.sampling_params.top_k=20 ++generate_params.offline.sampling_params.repetition_penalty=1.05 ++llm.prompt_template=qwen_chat_mt ++llm.exp_tag=Qwen3-4B-mt-zero ${THINKING} ++tokenize_args.enable_thinking=true ++evaluate.eval_func_args.check_num=0 ++evaluate.metrics_args.ignore_columns=[]"
# models["Qwen3-0.6B-mt-zero"]="/home/nfs05/shenyz/translation/verl/bs@128_n@8_m@1_@20251126_133125_@Qwen3-0.6B-mt-zero-data_17-20/global_step_400/actor/huggingface generate_params.offline.sampling_params.temperature=0.6 ++generate_params.offline.sampling_params.top_p=0.95 ++generate_params.offline.sampling_params.top_k=20 ++generate_params.offline.sampling_params.repetition_penalty=1.05 ++llm.prompt_template=qwen_chat_mt ++llm.exp_tag=Qwen3-0.6B-mt-zero ${THINKING} ++tokenize_args.enable_thinking=true ++evaluate.eval_func_args.check_num=0 ++evaluate.metrics_args.ignore_columns=[]"
# models["Qwen3-0.6B-mt-zero-mix-trained-2-stage"]="/home/nfs05/shenyz/llm_evaluate/mt_r1_zero_mix_trained generate_params.offline.sampling_params.temperature=0.6 ++generate_params.offline.sampling_params.top_p=0.95 ++generate_params.offline.sampling_params.top_k=20 ++generate_params.offline.sampling_params.repetition_penalty=1.05 ++llm.prompt_template=qwen_chat_mt ++llm.exp_tag=Qwen3-0.6B-mt-zero-mix-trained-stage2 ${THINKING} ++tokenize_args.enable_thinking=true ++evaluate.eval_func_args.check_num=0 ++evaluate.metrics_args.ignore_columns=[]"
# models["Qwen3-0.6B-mixed-plus-test"]="/home/nfs05/shenyz/translation/verl/bs@128_n@4_m@1_@20251202_122553_@Qwen3-0.6B-mix-plus-data_17-20_@4gpus/global_step_100/actor/huggingface generate_params.offline.sampling_params.temperature=0.6 ++generate_params.offline.sampling_params.top_p=0.95 ++generate_params.offline.sampling_params.top_k=20 ++llm.prompt_template=qwen_chat_mt ++llm.exp_tag=Qwen3-0.6B-mixed-plus-test ${THINKING} ++tokenize_args.enable_thinking=true ++evaluate.eval_func_args.check_num=1 ++evaluate.metrics_args.ignore_columns=[0]"
# models["Qwen3-0.6B-mixed-plus-test-2"]="/home/nfs05/shenyz/translation/verl/bs@128_n@4_m@1_@20251202_122553_@Qwen3-0.6B-mix-plus-data_17-20_@4gpus/global_step_200/actor/huggingface generate_params.offline.sampling_params.temperature=0.6 ++generate_params.offline.sampling_params.top_p=0.95 ++generate_params.offline.sampling_params.top_k=20 ++llm.prompt_template=qwen_chat_mt ++llm.exp_tag=Qwen3-0.6B-mixed-plus-test ${THINKING} ++tokenize_args.enable_thinking=true ++evaluate.eval_func_args.check_num=1 ++evaluate.metrics_args.ignore_columns=[0]"
# models["Qwen3-0.6B-mixed-plus-test-1-equal"]="/home/nfs06/shenyz/models/qwen3-0.6B-equal-tree-mix-en-zh generate_params.offline.sampling_params.temperature=0.6 ++generate_params.offline.sampling_params.top_p=0.95 ++generate_params.offline.sampling_params.top_k=20 ++llm.prompt_template=qwen_chat_mt ++llm.exp_tag=Qwen3-0.6B-mixed-plus-test-1-equal ${THINKING} ++tokenize_args.enable_thinking=true ++evaluate.eval_func_args.check_num=1 ++evaluate.metrics_args.ignore_columns=[0]"
# models["Qwen3-0.6B-trained-sft"]="/home/nfs05/shenyz/translation/verl/bs@128_n@8_m@2_@20251119_172057_@Qwen3-0.6B-sft/global_step_200/actor/huggingface/ generate_params.offline.sampling_params.temperature=0.6 ++generate_params.offline.sampling_params.top_p=0.95 ++generate_params.offline.sampling_params.top_k=20 ++generate_params.offline.sampling_params.repetition_penalty=1.05 ++llm.prompt_template=qwen_chat_mt ++llm.exp_tag=Qwen3-0.6B-trained-sft ${THINKING} ++tokenize_args.enable_thinking=false ++evaluate.eval_func_args.check_num=1"
# models["Qwen3-4B-trained-mixed"]="/home/nfs05/shenyz/translation/verl/bs@128_n@8_m@2_@20251122_100132_@Qwen3-4B-mix/global_step_200/actor/huggingface/ generate_params.offline.sampling_params.temperature=0.6 ++generate_params.offline.sampling_params.top_p=0.95 ++generate_params.offline.sampling_params.top_k=20 ++generate_params.offline.sampling_params.repetition_penalty=1.15 ++llm.prompt_template=qwen_chat_mt ++llm.exp_tag=Qwen3-4B-trained-mixed ${THINKING} ++tokenize_args.enable_thinking=true ++evaluate.eval_func_args.check_num=1"
# models["Qwen3-4B-trained-mixed-wmt-17-20"]="/home/nfs05/shenyz/translation/verl/bs@128_n@8_m@1_@20251123_214429_@Qwen3-4B-mix-data_17-20/global_step_200/actor/huggingface/ generate_params.offline.sampling_params.temperature=0.6 ++generate_params.offline.sampling_params.top_p=0.95 ++generate_params.offline.sampling_params.top_k=20 ++generate_params.offline.sampling_params.repetition_penalty=1.05 ++llm.prompt_template=qwen_chat_mt ++llm.exp_tag=Qwen3-4B-trained-mixed-wmt17-20 ${THINKING} ++tokenize_args.enable_thinking=true ++evaluate.eval_func_args.check_num=1"
# models["Qwen3-0.6B-trained-pow3-data-plus"]="/home/nfs05/shenyz/translation/verl/bs@256_n@8_m@3_@20251114_184242/global_step_300/actor/huggingface generate_params.offline.sampling_params.temperature=0.6 ++generate_params.offline.sampling_params.top_p=0.95 ++generate_params.offline.sampling_params.top_k=20 ++llm.prompt_template=qwen_chat_mt ${THINKING}"

LOG_DIR="./logs/main"
mkdir -p "$LOG_DIR"

MAX_RETRIES=3
task_idx=0
total_models=${#models[@]}
start_time_all=$(date +%s)

for model_name in "${!models[@]}"; do
    task_idx=$((task_idx + 1))
    model_path=$(echo "${models[$model_name]}" | awk '{print $1}')
    model_extra=$(echo "${models[$model_name]}" | cut -d' ' -f2-)
    start_time=$(date +%s)
    log_name="${LOG_DIR}/mt_${model_name}_trained_${THINKING}.log"

    echo "──────────────────────────────────────────────"
    echo "▶ [${task_idx}/${total_models}] Running $model_name"
    echo "   Log: $log_name"

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
