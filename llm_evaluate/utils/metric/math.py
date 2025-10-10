from __future__ import annotations

import re
from .registry import register
from .abstract import Metric


def extract_boxed_answer(response: str) -> str | None:
    """Extract the final answer from the model's response.

    The final answer is expected to be within a LaTeX \\boxed{} command.

    Args:
        response (str): The model's response string.

    Returns:
        str | None: The extracted answer, or None if no boxed answer is found.
    """
    match = re.search(r"\\boxed\{([^}]*)\}", response)
    if match:
        return match.group(1).strip()
    return None


@register("math_boxed_accuracy")
class MathBoxedAccuracy(Metric):
    """Metric that evaluates math responses based on LaTeX \\boxed{} answers."""

    def __call__(
        self,
        responses: list[str],
        references: list[str],
        extra_infos: list[dict] | None = None,
    ) -> dict:
        """
        Compute boxed accuracy metric for math problems.

        Args:
            responses (list[str]): Model outputs.
            references (list[str]): Gold solutions.
            extra_infos (list[dict] | None, optional): Unused here but kept for interface consistency.

        Returns:
            dict: {
                "score": list[float],
                "extra_dict": {
                    "extracted_answer": list[str | None]
                }
            }
        """
        from latex2sympy2_extended import NormalizationConfig
        from math_verify import LatexExtractionConfig, parse, verify, ExprExtractionConfig

        rewards: list[float] = []
        extracted_answers: list[str | None] = []

        for content, sol in zip(responses, references):
            gold_parsed = parse(sol)
            if len(gold_parsed) == 0:
                gold_parsed = parse(f"\\boxed{{{sol}}}")

            if len(gold_parsed) != 0:
                # Parse student response
                answer_parsed = parse(
                    content,
                    extraction_config=[
                        LatexExtractionConfig(
                            normalization_config=NormalizationConfig(
                                nits=False,
                                malformed_operators=False,
                                basic_latex=True,
                                boxed="all",
                                units=True,
                            ),
                            # Ensures that boxed is tried first
                            boxed_match_priority=0,
                            try_extract_without_anchor=True,
                        ),
                        ExprExtractionConfig(try_extract_without_anchor=False),
                    ],
                )

                if len(answer_parsed) == 0:
                    extracted_answers.append(None)
                else:
                    extracted_answers.append(answer_parsed[-1])

                try:
                    reward = float(verify(gold_parsed, answer_parsed))
                except Exception as e:
                    print(f"verify failed: {e}, answer: {answer_parsed}, gold: {gold_parsed}")
                    reward = 0
            else:
                # Gold solution is not parseable
                reward = 0
                print("Failed to parse gold solution: ", sol)

            rewards.append(reward)

        return {
            "score": rewards,
            "extra_dict": {
                "extracted_answer": extracted_answers,
            },
        }



import re
from verl.utils.reward_score.math_reward import compute_score as math_compute_score

def foramt_reward(solution_str):
    match = re.search(r"\\boxed\{([^}]*)\}", solution_str)
    if match:
        return True
    return False


def extract_final_answer(solution_str):
    matches = re.findall(r"The answer is:?\s*([^\n.]*)", solution_str)
    if matches:
        return matches[-1].strip()
    
    numbers = re.findall(r"(-?\d+(?:\.\d+)?)", solution_str) # fix "9. " will be extracted
    if numbers:
        return numbers[-1]
    
    return None

def check(solution_str, ground_truth):
    if math_compute_score(solution_str, ground_truth) > 0:
        return True
    else:
        if foramt_reward(solution_str) == False:
            soft_ans = f"\\boxed{{{extract_final_answer(solution_str)}}}"
            if math_compute_score(soft_ans, ground_truth) > 0:
                return True
    return False
    
def eval_score(
    solution_str, 
    ground_truth, 
    extra_info=None,
    data_source=None,
    format_reward=0.0,
    score=1.0
):
    if not foramt_reward(solution_str):
        format_reward = 0.0

    if check(solution_str, ground_truth):
        return score + format_reward

    return format_reward


@register("soft_math_accuracy")
class SoftMathAccuracy(Metric):

    def __call__(
        self,
        responses: list[str],
        references: list[str],
        extra_infos: list[dict] | None = None,
    ) -> dict:
        
        score = []
        for resp, ref in zip(responses, references):
            score.append(eval_score(resp, ref))

        return {"score": score}


@register("union_math_accuracy")
class UnionMathAccuracy(Metric):

    def __init__(self):
        self.math_boxed = MathBoxedAccuracy()
        self.soft_math = SoftMathAccuracy()

    def __call__(
        self,
        responses: list[str],
        references: list[str],
        extra_infos: list[dict] | None = None,
    ) -> dict:
        
        score1 = self.math_boxed(responses, references)["score"]
        score2 = self.soft_math(responses, references)["score"]

        return {"score": [max(x, y) for x, y in zip(score1, score2)]}