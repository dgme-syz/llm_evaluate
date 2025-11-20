from llm_evaluate.dataset import EvalDataset, register


@register("dummy")
class Dummydataset(EvalDataset):

    # https://github.com/fzp0424/MT-R1-Zero/blob/main/data/train/json/train_enzh_6565.jsonl
    def __init__(self, data_dir, subset_name=None, split="train", builder=None, extra_args=None):
        """
        Initialize the WMT24 dataset wrapper.

        Args:
            data_dir (str): Directory containing the dataset.
            subset_name (str, optional): Language pair, e.g., 'en-zh_CN'.
            split (str, optional): Dataset split to load ('train', 'test', etc.).
            builder (callable, optional): Custom dataset builder.
            extra_args (dict, optional): Additional arguments for dataset processing.
        """
        super().__init__(data_dir, subset_name, split, builder)


    def convert_item(self, examples, **kwargs):
        """
        Convert a single dataset example into structured format for LLM evaluation.

        Args:
            examples (dict): A single example from the dataset with 'source' and 'target' keys.

        Returns:
            dict: Structured item with prompt, ability, reward model, and extra info.
        """
        return examples
