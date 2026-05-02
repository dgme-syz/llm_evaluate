#!/usr/bin/env bash

set -e
set -o pipefail

ROOT_VERL_DIR="$(pwd)"

checkpoint_0="bs@128_n@8_m@1_@20251227_203107_@Qwen3-8B-main_en2fi_just_mean_kiwi_@8gpus"
checkpoint_1="bs@128_n@72_m@1_@20251227_203118_@Qwen3-8B-main_en2fi_mt-zero_@8gpus"

# Merge FSDP checkpoints
python -m verl.model_merger merge \
  --backend fsdp \
  --local_dir "${checkpoint_0}/global_step_400/actor/" \
  --target_dir "${checkpoint_0}/global_step_400/actor/huggingface"

python -m verl.model_merger merge \
  --backend fsdp \
  --local_dir "${checkpoint_1}/global_step_400/actor/" \
  --target_dir "${checkpoint_1}/global_step_400/actor/huggingface"

model_0="${ROOT_VERL_DIR}/${checkpoint_0}/global_step_400/actor/huggingface"
model_1="${ROOT_VERL_DIR}/${checkpoint_1}/global_step_400/actor/huggingface"

# Require bash 4+
if [[ "${BASH_VERSINFO:-0}" -lt 4 ]]; then
  echo "❌ This script requires bash 4+ (associative arrays unsupported)."
  exit 1
fi

declare -A models

models["ours-8b_en2fi"]="${model_0} \
generate_params.offline.sampling_params.temperature=0.6 \
++generate_params.offline.sampling_params.top_p=0.95 \
++generate_params.offline.sampling_params.top_k=20 \
++generate_params.offline.sampling_params.repetition_penalty=1.05 \
++llm.prompt_template=qwen_chat_mt \
++llm.exp_tag=ours-8b_en2fi"

models["mt-r1-zero_en2fi"]="${model_1} \
generate_params.offline.sampling_params.temperature=0.6 \
++generate_params.offline.sampling_params.top_p=0.95 \
++generate_params.offline.sampling_params.top_k=20 \
++generate_params.offline.sampling_params.repetition_penalty=1.05 \
++llm.prompt_template=qwen_chat_mt \
++evaluate.eval_func=base \
++evaluate.metrics_args.ignore_columns=[] \
++llm.exp_tag=mt-r1-zero_en2fi"

cd ../llm_evaluate

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

    # Safe fallback if THINKING not defined
    THINKING="${THINKING:-default}"

    log_name="${LOG_DIR}/mt_${model_name}_trained_${THINKING}.log"

    echo "──────────────────────────────────────────────"
    echo "▶ [${task_idx}/${total_models}] Running ${model_name}"
    echo "   Log: ${log_name}"

    attempt=0
    while (( attempt < MAX_RETRIES )); do
        attempt=$(( attempt + 1 ))
        echo "⏳ Attempt #${attempt}..."

        bash scripts/run_mt.sh \
            --path "${model_path}" \
            --log "${log_name}" \
            --extra "${model_extra}"

        exit_code=$?
        if [[ $exit_code -eq 0 ]]; then
            echo "✅ Success on attempt #${attempt}"
            break
        else
            echo "⚠️ Task failed with exit code ${exit_code}."
            if (( attempt < MAX_RETRIES )); then
                echo "🔁 Retrying..."
            else
                echo "❌ Max retries reached. Skipping..."
            fi
        fi
    done

    end_time=$(date +%s)
    duration=$(( end_time - start_time ))
    echo "✅ Finished: ${model_name} — took ${duration}s"
    echo
done

end_time_all=$(date +%s)
total_duration=$(( end_time_all - start_time_all ))

echo "🏁 All tasks completed in ${total_duration}s"
echo "──────────────────────────────────────────────"
