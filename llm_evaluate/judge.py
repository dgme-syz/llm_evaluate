import os
import json
import re
from typing import Optional, List

import hydra
from tqdm import tqdm
from omegaconf import DictConfig

from llm_evaluate.vllm.utils import build_model


def build_comparison_prompt(
    source: str,
    reference: str,
    translation_a: str,
    translation_b: str,
) -> str:
    """
    Build a reference-aware prompt for comparing two translations.
    """

    return f"""
You are an expert bilingual translation evaluator.

Your task is to compare two candidate translations against a human reference translation.

Evaluation Criteria:
1. Adequacy
2. Fluency
3. Terminology
4. Consistency with the reference
5. Overall quality

Important:
- Do NOT favor surface similarity to the reference.
- Focus on semantic equivalence.
- Avoid positional bias.

Output strictly one of:
A > B
B > A
A = B

Source:
{source}

Reference:
{reference}

Translation A:
{translation_a}

Translation B:
{translation_b}
""".strip()


def extract_scalar_reward(response: str) -> Optional[int]:
    """
    Extract decision from LLM output and convert to scalar reward.
    """

    match = re.search(r"\b(A\s*>\s*B|B\s*>\s*A|A\s*=\s*B)\b", response)
    if not match:
        return None

    decision = match.group(1).replace(" ", "")

    if decision == "A>B":
        return 1
    if decision == "B>A":
        return -1
    if decision == "A=B":
        return 0

    return None


def load_jsonl(path: str) -> List[dict]:
    """
    Load a JSONL file (one JSON object per line).
    """
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data


@hydra.main(config_path="config", config_name="judge", version_base=None)
def main_judge(cfg: DictConfig) -> None:
    """
    Pairwise LLM judge for two JSONL files.
    """

    llm = build_model(cfg)
    print(f"Judge model: {cfg.server.model}")

    data_list = getattr(cfg, "data", None)
    if data_list is None or len(data_list) != 2:
        raise ValueError("cfg.data must contain exactly two JSONL file paths.")

    save_outputs = getattr(cfg, "save_outputs", False)
    outputs_dir = getattr(cfg, "outputs_dir", "./outputs")

    if save_outputs:
        os.makedirs(outputs_dir, exist_ok=True)
        output_path = os.path.join(outputs_dir, "judge_outputs.jsonl")
        print(f"Saving judge outputs to: {output_path}")
        fout = open(output_path, "w", encoding="utf-8")
    else:
        fout = None

    # Load JSONL files
    data0 = load_jsonl(data_list.x)
    data1 = load_jsonl(data_list.y)

    if len(data0) != len(data1):
        raise ValueError("The two JSONL files must have the same number of lines.")

    scores_a = 0
    scores_b = 0
    ties_or_unparsed = 0

    for x, y in tqdm(zip(data0, data1), total=len(data0)):

        src_x = x["extra_info"]["src"]
        src_y = y["extra_info"]["src"]

        if src_x != src_y:
            raise ValueError("Source mismatch between paired files.")

        reference = x["reward_model"]["ground_truth"]

        # response may be list or string
        trans_a = x["response"][0] if isinstance(x["response"], list) else x["response"]
        trans_b = y["response"][0] if isinstance(y["response"], list) else y["response"]

        judge_prompt = build_comparison_prompt(
            source=src_x,
            reference=reference,
            translation_a=trans_a,
            translation_b=trans_b,
        )

        judge_response = llm.generate(judge_prompt)
        judge_res = extract_scalar_reward(judge_response)

        if judge_res == 1:
            scores_a += 1
            decision = "A>B"
        elif judge_res == -1:
            scores_b += 1
            decision = "B>A"
        elif judge_res == 0:
            ties_or_unparsed += 1
            decision = "A=B"
        else:
            ties_or_unparsed += 1
            decision = None

        # Save JSONL output
        if fout is not None:
            record = {
                "source": src_x,
                "reference": reference,
                "translation_a": trans_a,
                "translation_b": trans_b,
                "judge_output": judge_response,
                "parsed_decision": decision,
                "scalar_reward": judge_res,
            }
            fout.write(json.dumps(record, ensure_ascii=False) + "\n")

    if fout is not None:
        fout.close()

    total = len(data0)

    print(f"Total samples: {total}")
    print(f"A wins: {scores_a}")
    print(f"B wins: {scores_b}")
    print(f"Ties / unparsed: {ties_or_unparsed}")
    print(f"A win rate: {scores_a / total:.4f}")
    print(f"B win rate: {scores_b / total:.4f}")


if __name__ == "__main__":
    main_judge()
