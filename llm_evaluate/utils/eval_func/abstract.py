from abc import ABC, abstractmethod


class EvalFunc(ABC):
    def __init__(self, llm):
        self.llm = llm
    
    def __call__(self, examples):
        return self.evaluate(examples)

    @abstractmethod
    def evaluate(self, examples):
        pass    
