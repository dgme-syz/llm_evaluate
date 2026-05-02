from llm_evaluate.dataset import EvalDataset, register


@register("dummy")
class Dummydataset(EvalDataset):
    """Pass-through dataset that returns each row unchanged.

    Useful for quick smoke tests where the dataset already matches the expected
    pipeline schema and no prompt formatting is needed.
    """

    def __init__(self, data_dir, subset_name=None, split="train", builder=None, extra_args=None):
        """Initialize the dummy dataset wrapper. See ``EvalDataset`` for argument semantics."""
        super().__init__(data_dir, subset_name, split, builder)

    def convert_item(self, examples, **kwargs):
        """Return the example unchanged."""
        return examples
