set -x

export TOKENIZERS_PARALLELISM=false

CUDA_VISIBLE_DEVICES=2,3 python3 -m llm_evaluate.main_eval