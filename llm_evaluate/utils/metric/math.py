from .registry import register

def extract_boxed_answer(response: str) -> str:
    """Extract the final answer from the model's response.

    The final answer is expected to be within a LaTeX \\boxed{} command.

    Args:
        response (str): The model's response string.

    Returns:
        str: The extracted answer, or None if no boxed answer is found.
    """
    import re

    match = re.search(r"\\boxed\{([^}]*)\}", response)
    if match:
        return match.group(1).strip()
    return None


@register("math_boxed_accuracy")
def math_boxed_accuracy(responses, references, extra_info=None):
    from latex2sympy2_extended import NormalizationConfig
    from math_verify import LatexExtractionConfig, parse, verify, ExprExtractionConfig

    rewards = []
    extracted_answers = []

    for content, sol in zip(responses, references):
        gold_parsed = parse(
            sol,
            extraction_mode="first_match",
        )
        if len(gold_parsed) != 0:
            # We require the answer to be provided in correct latex (no malformed operators)
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
                extracted_answers += [None]
            else:
                extracted_answers += [answer_parsed[-1]]

            # Compute binary rewards if verifiable, `None` otherwise to skip this example
            try:
                reward = float(verify(gold_parsed, answer_parsed))
            except Exception as e:
                print(f"verify failed: {e}, answer: {answer_parsed}, gold: {gold_parsed}")
                reward = 0
        else:
            # If the gold solution is not parseable, we assign `None` to skip this example
            reward = 0
            print("Failed to parse gold solution: ", sol)
        rewards.append(reward)

    return {
        "score": rewards,
        "extra_dict": {
            "extracted_answer": extracted_answers,
        }
    }



