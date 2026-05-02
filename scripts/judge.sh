python -m llm_evaluate.judge \
  data.x=/home/nfs05/shenyz/llm_evaluate/outputs/test/x.jsonl \
  data.y=/home/nfs05/shenyz/llm_evaluate/outputs/test/y.jsonl \
  server.model="deepseek-v3.2" \
  server.openai_args.base_url="https://api.bltcy.ai/v1/" \
  server.openai_args.api_key="sk-REDACTED"
