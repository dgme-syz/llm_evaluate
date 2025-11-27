import json

jsonl_path = "outputs/trained/Qwen3-4B-trained-mixed/flores_flores_en_flores_zh_recheck_check_num_1_use_thinking_true_use_thinking_prompts_false_.jsonl"
hyp_path = "hyp.txt"
ref_path = "ref.txt"

with open(jsonl_path, "r", encoding="utf-8") as f_jsonl, \
     open(hyp_path, "w", encoding="utf-8") as f_hyp, \
     open(ref_path, "w", encoding="utf-8") as f_ref:
    
    for line in f_jsonl:
        if not line.strip():
            continue
        data = json.loads(line)
        
        # 提取 response 的第二条翻译作为 hyp
        response_list = data.get("response", [])
        if not isinstance(response_list, list):
            response_list = [response_list]
        hyp = response_list[-1].replace("\n", " ").strip()
        
        # 提取 ground_truth 作为 ref
        ref = data.get("reward_model", {}).get("ground_truth", "").replace("\n", " ").strip()
        if not ref:
            continue
        
        f_hyp.write(hyp + "\n")
        f_ref.write(ref + "\n")

print(f"Hypotheses written to {hyp_path}, references written to {ref_path}")
