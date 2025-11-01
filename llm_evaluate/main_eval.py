import os
import gc
import json
from typing import Any
from typing import List

import torch
import hydra
import numpy as np
from hydra import compose
from tqdm import tqdm
from omegaconf import DictConfig, OmegaConf, ListConfig

from llm_evaluate.dataset import get_dataset
from llm_evaluate.vllm.utils import build_model
from llm_evaluate.utils.metric import get_metrics
from llm_evaluate.utils.eval_func import get_eval
from llm_evaluate.utils.merge_wrapper import decorator

def resolve_data_list_with_hydra(data_list: List[str]) -> List[DictConfig]:
    """
    Resolve dataset paths into DictConfig objects using Hydra's compose API.

    Args:
        data_list: List of strings like "translation/challenge_set_en2zh"
        config_path: Base path to Hydra config files
    Returns:
        List of DictConfig
    """
    resolved = []
    for item in data_list:
        if isinstance(item, str):
            # item is something like "translation/challenge_set_en2zh"
            cfg_name = item
            cfg_obj = compose(config_name=cfg_name)
            while isinstance(cfg_obj, DictConfig) and len(cfg_obj) == 1 and isinstance(list(cfg_obj.values())[0], DictConfig):
                cfg_obj = list(cfg_obj.values())[0]
            resolved.append(cfg_obj)
        else:
            resolved.append(item)
    return resolved

def compute_metrics_save_results(
    data_cfg: DictConfig,
    data: Any,
    metric_funcs: dict[str, Any],
    eval_cfg: DictConfig,
    cfg: DictConfig, 
):
    answers = [x["reward_model"]["ground_truth"] for x in data]
    # in lower version's Datasets, Dataset use Column
    # in newer version's Datasets, Dataset use list
    # So we use list() to ensure it can work in a lower version
    responses: list[list[str]] = list(data["response"])
    extras = list(data["extra_info"])

    # 9. Compute metrics
    merge_strategy = eval_cfg.get("merge_strategy", "pass")
    num_generations = (
        cfg.sample_params.online.get("n")
        if cfg.get("use_server", False)
        else cfg.generate_params.offline.sampling_params.get("n", 1)
    )
    print(OmegaConf.to_yaml(data_cfg))
    print("Computing metrics...")
    for name, func in metric_funcs.items():
        scores: dict[str, Any] = {}
        decorated_func = decorator(func, merge_strategy=merge_strategy)
        sub_score = decorated_func(responses, answers, extras)

        # Ensure sub_score is valid
        if not isinstance(sub_score, dict) or "score" not in sub_score:
            raise ValueError(f"Metric '{name}' must return a dict with key 'score', got: {type(sub_score)}")

        score_value = sub_score["score"]

        # --- Handle extra_dict (detailed metrics) ---
        extra_dict = sub_score.get("extra_dict", {})
        for k, v in extra_dict.items():
            if hasattr(v, "__len__") and len(v) != len(data):
                raise ValueError(f"Length mismatch for extra_dict '{k}' (expected {len(data)}, got {len(v)})")

            data = data.add_column(f"{data_cfg.name}_{name}_{k}", v)
            try:
                scores[f"{name}_list_{k}_score"] = np.mean(v, axis=0)
            except Exception as e:
                print(f"[WARN] Cannot compute mean for '{k}': {e}, got {v[:10]}")

        # --- Handle score itself ---
        if isinstance(score_value, list) and all(isinstance(x, (int, float)) for x in score_value):
            data = data.add_column(
                f"{data_cfg.name}_{name}_score_{merge_strategy}@{num_generations}", score_value
            )
            scores[f"{name}_score"] = float(np.mean(score_value))
        else:
            scores[f"{name}_score"] = score_value

        print(scores)
    torch.cuda.empty_cache()

    print("Evaluation finished.")

    # 10. Save evaluation outputs
    if cfg.get("save_outputs", False):
        out_dir = cfg.get("outputs_dir", "./output")
        os.makedirs(out_dir, exist_ok=True)

        if cfg.get("use_server", False):
            model_name = cfg.server.model
        else:
            model_name = cfg.llm.vllm.model.split("/")[-1] 

        filename = (
            f"{data_cfg.subset_name}_{data_cfg.name}_"
            f"{model_name}_"
            f"{eval_cfg.get('eval_func')}"
            f"{str(eval_cfg.get('eval_func_args'))}.jsonl"
        )
        out_path = os.path.join(out_dir, filename)

        with open(out_path, "w", encoding="utf-8") as f:
            for item in tqdm(data, desc="Saving outputs"):
                json.dump(item, f, ensure_ascii=False, indent=4)
                f.write("\n")

        print(f"Saved outputs to: {out_path}")


@hydra.main(config_path="config", config_name="config", version_base=None)
def main_evaluate(cfg: DictConfig) -> None:
    """Main entry point for model evaluation using multiple datasets and metrics."""
    
    print(f"Loaded config:\n{OmegaConf.to_yaml(cfg)}")

    # 1. Initialize model
    llm = build_model(cfg)
    print("Model initialized.")

    # 2. Prepare evaluation configuration and metrics
    eval_cfg = getattr(cfg, "evaluate", {})
    eval_func = get_eval(eval_cfg.get("eval_func"))(llm, **eval_cfg.eval_func_args)

    # 3. Ensure cfg.data is a list for uniform processing
    cfg_data_raw = cfg.data if isinstance(cfg.data, ListConfig) else ListConfig([cfg.data])
    cfg_data = resolve_data_list_with_hydra(cfg_data_raw)
    print(cfg_data)

    # 4. Iterate over datasets
    middle_states = []
    for i, data_cfg in enumerate(cfg_data):
        subset_name = getattr(data_cfg, "subset_name", None)
        if cfg.get("use_server", False):
            backup_args = llm.sampling_params.copy()
            if "generate_args" in data_cfg:
                llm.merge_data_args(data_cfg["generate_args"])
    
        print(f"Evaluating dataset: {data_cfg.name}-{subset_name}")

        dataset_cls = get_dataset(data_cfg.name)
        if cfg.get("use_server", False):
            model_name = cfg.server.model
            prompt_type = cfg.server.get("prompt_template", None)
        else:
            model_name = cfg.llm.vllm.model.split("/")[-1] 
            prompt_type = cfg.llm.get("prompt_template", None)
        dataset = dataset_cls(
            data_dir=data_cfg.data_path,
            subset_name=subset_name,
            split=getattr(data_cfg, "split", "test"),
            builder=getattr(data_cfg, "builder", None),
            extra_args={
                "model": model_name,
                "prompt_template": prompt_type,
                **data_cfg.get("dataset_args"),
            }
        )

        # 5. Preprocess dataset
        data = dataset.map(
            batched=False,
            save_columns=getattr(data_cfg, "save_columns", False)
        )

        # 6. Subsample for debugging if requested
        if (num_samples := getattr(data_cfg, "num_samples", None)):
            data = data.select(range(num_samples))
            print(f"[Dev] Using num_samples={num_samples} for debugging.")

        # 7. Prepare evaluation function
        batch_size = eval_cfg.get("val_size", 1)
        

        # 8. Apply evaluation function to dataset
        data = data.map(
            lambda ex: eval_func(ex),
            batched=True,
            batch_size=batch_size,
            drop_last_batch=False,
            load_from_cache_file=False,
        )

        middle_states.append(
            (data_cfg, data, eval_cfg, cfg)
        )
        if cfg.get("use_server", False):
            llm.update_sampling_params(backup_args)
        
    if not cfg.get("use_server", False):
        llm.llm.sleep(level=2)
    metric_funcs = get_metrics(eval_cfg.get("metrics", []))
    print(f"Loaded metrics: {list(metric_funcs.keys())}")

    for data_cfg, data, eval_cfg, cfg in middle_states:
        compute_metrics_save_results(
            data_cfg,
            data,
            metric_funcs,
            eval_cfg,
            cfg
        )





if __name__ == "__main__":
    main_evaluate()
