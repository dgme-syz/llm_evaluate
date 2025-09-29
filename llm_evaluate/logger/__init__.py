from abc import ABC, abstractmethod


class BaseLogger(ABC):

    def __init__(self, config, dataset):
        self.config = config
        self.dataset = dataset

    def log(self): 
        raise NotImplemented("need to be finished by the user.")
    