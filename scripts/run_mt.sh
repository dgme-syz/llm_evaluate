set -x

export TOKENIZERS_PARALLELISM=false

CUDA_VISIBLE_DEVICES=1,2 python3 -m llm_evaluate.main_eval --config-name config_mt