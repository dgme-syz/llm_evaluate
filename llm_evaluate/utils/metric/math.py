from __future__ import annotations

import re
from .registry import register
from .abstract import Metric


# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2022 EleutherAI and the HuggingFace Inc. team. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# Adapted from https://github.com/EleutherAI/lm-evaluation-harness/blob/main/lm_eval/tasks/hendrycks_math/utils.py


def math_compute_score(solution_str, ground_truth) -> float:
    retval = 0.0
    try:
        string_in_last_boxed = last_boxed_only_string(solution_str)
        if string_in_last_boxed is not None:
            answer = remove_boxed(string_in_last_boxed)
            if is_equiv(answer, ground_truth):
                retval = 1.0
    except Exception as e:
        print(e)

    return retval


# string normalization from https://github.com/EleutherAI/lm-evaluation-harness/blob/master/lm_eval/tasks/hendrycks_math.py
def is_equiv(str1, str2, verbose=False):
    if str1 is None and str2 is None:
        print("WARNING: Both None")
        return True
    if str1 is None or str2 is None:
        return False

    try:
        ss1 = strip_string(str1)
        ss2 = strip_string(str2)
        if verbose:
            print(ss1, ss2)
        return ss1 == ss2
    except Exception:
        return str1 == str2


def remove_boxed(s):
    if "\\boxed " in s:
        left = "\\boxed "
        assert s[: len(left)] == left
        return s[len(left) :]

    left = "\\boxed{"

    assert s[: len(left)] == left
    assert s[-1] == "}"

    return s[len(left) : -1]


def last_boxed_only_string(string):
    idx = string.rfind("\\boxed")
    if "\\boxed " in string:
        return "\\boxed " + string.split("\\boxed ")[-1].split("$")[0]
    if idx < 0:
        idx = string.rfind("\\fbox")
        if idx < 0:
            return None

    i = idx
    right_brace_idx = None
    num_left_braces_open = 0
    while i < len(string):
        if string[i] == "{":
            num_left_braces_open += 1
        if string[i] == "}":
            num_left_braces_open -= 1
            if num_left_braces_open == 0:
                right_brace_idx = i
                break
        i += 1

    retval = None if right_brace_idx is None else string[idx : right_brace_idx + 1]

    return retval


def fix_fracs(string):
    substrs = string.split("\\frac")
    new_str = substrs[0]
    if len(substrs) > 1:
        substrs = substrs[1:]
        for substr in substrs:
            new_str += "\\frac"
            if substr[0] == "{":
                new_str += substr
            else:
                try:
                    assert len(substr) >= 2
                except:  # noqa: E722
                    return string
                a = substr[0]
                b = substr[1]
                if b != "{":
                    if len(substr) > 2:
                        post_substr = substr[2:]
                        new_str += "{" + a + "}{" + b + "}" + post_substr
                    else:
                        new_str += "{" + a + "}{" + b + "}"
                else:
                    if len(substr) > 2:
                        post_substr = substr[2:]
                        new_str += "{" + a + "}" + b + post_substr
                    else:
                        new_str += "{" + a + "}" + b
    string = new_str
    return string


def fix_a_slash_b(string):
    if len(string.split("/")) != 2:
        return string
    a = string.split("/")[0]
    b = string.split("/")[1]
    try:
        a = int(a)
        b = int(b)
        assert string == "{}/{}".format(a, b)
        new_string = "\\frac{" + str(a) + "}{" + str(b) + "}"
        return new_string
    except:  # noqa: E722
        return string


def remove_right_units(string):
    # "\\text{ " only ever occurs (at least in the val set) when describing units
    if "\\text{ " in string:
        splits = string.split("\\text{ ")
        assert len(splits) == 2
        return splits[0]
    else:
        return string


def fix_sqrt(string):
    if "\\sqrt" not in string:
        return string
    splits = string.split("\\sqrt")
    new_string = splits[0]
    for split in splits[1:]:
        if split[0] != "{":
            a = split[0]
            new_substr = "\\sqrt{" + a + "}" + split[1:]
        else:
            new_substr = "\\sqrt" + split
        new_string += new_substr
    return new_string


def strip_string(string):
    # linebreaks
    string = string.replace("\n", "")

    # remove inverse spaces
    string = string.replace("\\!", "")

    # replace \\ with \
    string = string.replace("\\\\", "\\")

    # replace tfrac and dfrac with frac
    string = string.replace("tfrac", "frac")
    string = string.replace("dfrac", "frac")

    # remove \left and \right
    string = string.replace("\\left", "")
    string = string.replace("\\right", "")

    # Remove circ (degrees)
    string = string.replace("^{\\circ}", "")
    string = string.replace("^\\circ", "")

    # remove dollar signs
    string = string.replace("\\$", "")

    # remove units (on the right)
    string = remove_right_units(string)

    # remove percentage
    string = string.replace("\\%", "")
    string = string.replace("\%", "")  # noqa: W605

    # " 0." equivalent to " ." and "{0." equivalent to "{." Alternatively, add "0" if "." is the start of the string
    string = string.replace(" .", " 0.")
    string = string.replace("{.", "{0.")
    # if empty, return empty string
    if len(string) == 0:
        return string
    if string[0] == ".":
        string = "0" + string

    # to consider: get rid of e.g. "k = " or "q = " at beginning
    if len(string.split("=")) == 2 and len(string.split("=")[0]) <= 2:
        string = string.split("=")[1]

    # fix sqrt3 --> sqrt{3}
    string = fix_sqrt(string)

    # remove spaces
    string = string.replace(" ", "")

    # \frac1b or \frac12 --> \frac{1}{b} and \frac{1}{2}, etc. Even works with \frac1{72} (but not \frac{72}1).
    # Also does a/b --> \\frac{a}{b}
    string = fix_fracs(string)

    # manually change 0.5 --> \frac{1}{2}
    if string == "0.5":
        string = "\\frac{1}{2}"

    # NOTE: X/Y changed to \frac{X}{Y} in dataset, but in simple cases fix in case the model output is X/Y
    string = fix_a_slash_b(string)

    return string


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