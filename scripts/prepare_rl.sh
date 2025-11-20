#!/usr/bin/env bash
set -euo pipefail

INPUT_FILE="/home/nfs05/shenyz/llm_evaluate/outputs/sft/challenge_set_en2zh_deepseek-reasoner_recheck_qwen3_0.6b.jsonl"
OUTPUT_PATH=/home/nfs06/shenyz/data/recheck_sft/
NAME=dpsk-3.2-exp-qwen3-0.6b-challenge-set-en2zh-sft
TEMPLATE_TYPE="recheck"
TEMPLATE_ARGS='{"thinking": false, "sft": true}'

python3 scripts/data_process.py \
  --file "$INPUT_FILE" \
  --template_type "$TEMPLATE_TYPE" \
  --template_args "$TEMPLATE_ARGS" \
  --save_dir "$OUTPUT_PATH" \
  --name "$NAME"

echo "[INFO] Preprocessing completed successfully."
