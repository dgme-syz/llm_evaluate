import os

import hydra
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

from llm_evaluate.dataset import get_dataset
from llm_evaluate.vllm.utils import build_model
from llm_evaluate.utils.metric import get_metrics
import torch
import gc
from vllm.distributed.parallel_state import cleanup_dist_env_and_memory


def generate_batched(examples, llm):
    """
    Batch inference function:
    - Input: examples containing "prompt"
    - Output: updated prompts with assistant responses appended
    """
    batched_prompts = examples["prompt"]
    batched_responses = llm.generate(batched_prompts)

    new_prompts = [
        prompt + [{"role": "assistant", "content": resp}]
        for prompt, resp in zip(batched_prompts, batched_responses)
    ]


    return {"prompt": new_prompts, "response": batched_responses}


@hydra.main(config_path=".", config_name="config")
def main_evaluate(config: DictConfig):
    print("Loaded config:\n", OmegaConf.to_yaml(config))

    # -----------------------------
    # 1. Prepare dataset
    # -----------------------------
    dataset_name = config.data.name
    data_path = config.data.data_path
    subset_name = getattr(config.data, "subset_name", None)
    split = getattr(config.data, "split", "test")
    builder = getattr(config.data, "builder", None)

    dataset_cls = get_dataset(dataset_name)
    dataset_obj = dataset_cls(data_path, subset_name=subset_name, split=split, builder=builder)
    data = dataset_obj.map(batched=False)
    
    if config.data.get("num_samples", None) is not None:
        data = data.select(range(config.data.num_samples))
        print(f"[Dev] Note: using num_samples={config.data.num_samples} for debugging.")

    # -----------------------------
    # 2. Initialize 
    # -----------------------------
    evaluate_config = getattr(config, "evaluate", {})
    batch_size = evaluate_config.get("val_size", 1)
    eval_func = get_metrics(evaluate_config.get("metrics"))


    # -----------------------------
    # 3. Run batched generation
    # -----------------------------
    llm = build_model(config)
    print("Model initialized.")

    data = data.map(
        lambda examples: generate_batched(examples, llm),
        batched=True,
        batch_size=batch_size,
        drop_last_batch=False,
    )

    answers = [x["reward_model"]["ground_truth"] for x in data]
    responses = data["response"]

    score = {}
    for metric_name, func in eval_func.items():
        sub_score = func(responses, answers, data[0].get("extra_info", {}))

        # score may be for the whole or each example
        if isinstance(sub_score, list):
            data = data.add_column(f"{metric_name}_score", sub_score)

        score[metric_name] = sub_score
        print(f"{metric_name}: {score[metric_name]:.4f}")
    print("Evaluation finished.")

    if config.get("save_outputs", False):
        import json
        
        file_name = f"{config.get('outputs_dir', './output')}/{dataset_name}_{config.llm.model.replace('/', '_')}_{dataset_name}.jsonl"
    
        if not os.path.exists(os.path.dirname(file_name)):
            os.makedirs(os.path.dirname(file_name), exist_ok=True)

        with open(file_name, "w") as f:
            for item in tqdm(data, desc="Saving outputs"):
                json_line = json.dumps(item, ensure_ascii=False, indent=4)
                f.write(json_line + "\n")

    del llm
    gc.collect()

    if torch.distributed.is_initialized():
        cleanup_dist_env_and_memory(True)
if __name__ == "__main__":
    main_evaluate()
