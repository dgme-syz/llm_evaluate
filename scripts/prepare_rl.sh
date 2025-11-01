#!/usr/bin/env bash
set -euo pipefail

INPUT_FILE="outputs/mt_thinking/en-zh_CN_train_mixed_Qwen3-8B_recheck{'check_num': 0, 'use_thinking': 'false_mixed_data'}.jsonl"
OUTPUT_PATH=/home/nfs06/shenyz/data/recheck/
NAME=Qwen3-8B_mixed_en_zh
TEMPLATE_TYPE="recheck"
TEMPLATE_ARGS='{"thinking": false}'

python3 scripts/data_process.py \
  --file "$INPUT_FILE" \
  --template_type "$TEMPLATE_TYPE" \
  --template_args "$TEMPLATE_ARGS" \
  --save_dir "$OUTPUT_PATH" \
  --name "$NAME"

echo "[INFO] Preprocessing completed successfully."
