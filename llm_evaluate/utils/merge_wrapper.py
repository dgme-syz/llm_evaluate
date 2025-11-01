from typing import Sequence, Literal

from .metric.abstract import Metric
from llm_evaluate.vllm.cls import flatten_with_paths, reconstruct_from_paths

def decorator(obj: Metric, merge_strategy: Literal["avg", "pass", "total"] = "pass"):

    def wrapper(responses: Sequence[Sequence[str]], answers: Sequence[str], extra_infos: Sequence[dict]):
        ret = {}

        if merge_strategy == "total":
            paths, all_responses = zip(*flatten_with_paths(responses))
            all_answers = [ans for ans, resps in zip(answers, responses) for _ in resps]
            all_extra_infos = [info for info, resps in zip(extra_infos, responses) for _ in resps]
            res = obj(all_responses, all_answers, all_extra_infos)

            if "score" not in res:
                raise ValueError(f"Metric must have a single 'score' key. Got {res.keys()} instead.")

            if isinstance(res["score"], (int, float)):
                ret.update({"score": res["score"]})
                tmp = res.get("extra_dict", {})
                for k, v in tmp.items():
                    tmp[k] = reconstruct_from_paths(paths, v)
                ret.update({"extra_dict": tmp})
            else:
                raise NotImplementedError("Metric returning list is not supported for 'total' strategy.")

        else:
            merge_dict = {}
            for j, row_responses in enumerate(responses):
                res = obj(row_responses, [answers[j]] * len(row_responses), extra_infos)
                for k, v in res.items():
                    merge_dict.setdefault(k, []).append(v)

            if merge_strategy == "pass":
                results = [max(x) for x in merge_dict["score"]]
            elif merge_strategy == "avg":
                results = [
                    (sum(x) / len(x)) if (x and all(isinstance(i, (int, float)) for i in x)) else None
                    for x in merge_dict["score"]
                ]
            else:
                raise NotImplementedError(f"Unsupported merge_strategy: {merge_strategy}")

            scores_per_response = merge_dict.pop("score")
            ret = {
                "score": results,
                "extra_dict": {
                    "scores_per_response": scores_per_response,
                    **merge_dict,
                }
            }

        return ret

    return wrapper


        

