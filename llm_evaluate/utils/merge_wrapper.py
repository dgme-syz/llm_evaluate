from typing import Sequence, Literal

from .metric.abstract import Metric


def decorator(obj: Metric, merge_strategy: Literal["avg", "pass", "total"] = "pass"):

    def wrapper(
        responses: Sequence[Sequence[str]], 
        answers: Sequence[str], 
        extra_infos: Sequence[dict], 
    ):
        ret = {}

        if merge_strategy == "total":
            all_responses = [resp for resps in responses for resp in resps]
            all_answers = [ans for ans, res in zip(answers, responses) for _ in res]
            results = obj(all_responses, all_answers, extra_infos)["score"]
        else:
            m = len(responses)
            scores = [[] for _ in range(m)]
            results = []

            for j in range(m):  
                row_responses = responses[j]
                scores[j] = obj(row_responses, [answers[j]] * len(row_responses), extra_infos)["score"]

            detailed_results = scores
            n = len(responses[0])
            if merge_strategy == "pass":
                results = [max(x) for x in scores]
            elif merge_strategy == "avg":
                results = [sum(x) / len(x) if len(x) else None for x in scores]
            else:
                raise NotImplementedError(
                    f"Please select merge_strategy from ['avg', 'pass', 'total']."
                    f"But got {merge_strategy}."
                )    
            
            ret.update({
                "extra_dict": {
                    "pass@1_score": [x[0] if len(x) else None for x in scores],
                    "detail_results": detailed_results,
                }
            })
            
        ret.update({"score": results})
        return ret

    return wrapper


        

