import hydra
from omegaconf import DictConfig, OmegaConf

from llm_evaluate.dataset.registry import get_dataset
from llm_evaluate.vllm.utils import build_model
from llm_evaluate.utils.metric.registry import get_metrics

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

    print("\nSample data:", data[:3])

    # -----------------------------
    # 2. Initialize 
    # -----------------------------
    evaluate_config = getattr(config, "evaluate", {})
    batch_size = evaluate_config.get("val_size", 1)
    eval_func = get_metrics(evaluate_config.get("metrics"))

    for metric_name, func in eval_func.items():
        print(f"Using evaluation metric: {metric_name}")
        print(f"Sample evaluation result: {func(['This is a test.', 'hello'], ['This is a test.', 'we'], {'tgt_lang': 'en'})}")



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
        score[metric_name] = func(responses, answers, data[0].get("extra_info", {}))
        print(f"{metric_name}: {score[metric_name]:.4f}")
    print("Evaluation finished.")


if __name__ == "__main__":
    main_evaluate()
