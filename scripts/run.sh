set -x

export TOKENIZERS_PARALLELISM=false

CUDA_VISIBLE_DEVICES=0,1 python3 -m llm_evaluate.main_eval