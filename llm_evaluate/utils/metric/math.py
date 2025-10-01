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
            gold_parsed = parse(sol, extraction_mode="first_match")

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
                        ExprExtractionConfig(),
                    ],
                    extraction_mode="first_match",
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
