import os

import hydra
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

import torch
import gc
from vllm.distributed.parallel_state import cleanup_dist_env_and_memory

from llm_evaluate.dataset import get_dataset
from llm_evaluate.vllm.utils import build_model
from llm_evaluate.utils.metric import get_metrics
from llm_evaluate.utils.eval_func import get_eval
from llm_evaluate.utils.merge_wrapper import decorator

@hydra.main(config_path=".", config_name="config", version_base=None)
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
    save_columns = getattr(config.data, "save_columns", False)

    dataset_cls = get_dataset(dataset_name)
    dataset_obj = dataset_cls(data_path, subset_name=subset_name, split=split, builder=builder)
    data = dataset_obj.map(batched=False, save_columns=save_columns)

    if config.data.get("num_samples", None) is not None:
        data = data.select(range(config.data.num_samples))
        print(f"[Dev] Note: using num_samples={config.data.num_samples} for debugging.")

    # -----------------------------
    # 2. Initialize 
    # -----------------------------
    evaluate_config = getattr(config, "evaluate", {})
    batch_size = evaluate_config.get("val_size", 1)
    metric_func = get_metrics(evaluate_config.get("metrics"))


    # -----------------------------
    # 3. Run batched generation
    # -----------------------------
    llm = build_model(config)
    print("Model initialized.")
    eval_func_cls = get_eval(evaluate_config.get("eval_func"))
    eval_func = eval_func_cls(llm)
    
    data = data.map(
        lambda examples: eval_func(examples),
        batched=True,
        batch_size=batch_size,
        drop_last_batch=False,
        load_from_cache_file=False
    )
    
    if config.get("use_server", False) == False:
        llm.llm.sleep(level=2)
    

    answers = [x["reward_model"]["ground_truth"] for x in data]
    responses: list[list[str]] = data["response"]
    extra_infos = data["extra_info"]

    score = {}

    merge_strategy = evaluate_config.get("merge_strategy", "pass")
    
    if config.use_server:
        num_generations = config.sample_params.online.get("n")
    else:
        num_generations = config.sample_params.offline.get("n")

    for metric_name, func in metric_func.items():
        func = decorator(func, merge_strategy=merge_strategy)
        sub_score = func(responses, answers, extra_infos)

        if isinstance(sub_score, dict) and "score" in sub_score:
            # for metrics that return dict with "score" and "extra_dict"
            extra_dict = sub_score.get("extra_dict", {})
            for k, v in extra_dict.items():
                assert len(v) == len(data), f"Length of extra_dict {k} does not match data length."
                data = data.add_column(f"{config.data.name}_{metric_name}_{k}", v)
            sub_score = sub_score["score"]

        if isinstance(sub_score, list) and all(isinstance(x, (int, float)) for x in sub_score):
            data = data.add_column(f"{config.data.name}_{metric_name}_score_{merge_strategy}@{num_generations}", sub_score)
            score[metric_name] = sum(sub_score) / len(sub_score)
        else:
            score[metric_name] = sub_score
        
        if isinstance(score[metric_name], float):
            print(f"{metric_name}: {score[metric_name]:.4f}")

    print("Evaluation finished.")

    if config.get("save_outputs", False):
        import json

        file_name = f"{config.get('outputs_dir', './output')}/{config.exp_name_prefix}_{dataset_name}_{config.llm.model.replace('/', '_')}_{dataset_name}_{evaluate_config.get('eval_func')}.jsonl"

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
