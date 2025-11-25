import multiprocessing
import os
from typing import Any
import subprocess

import ray


class ClassWithInitArgs:
    def __init__(self, cls, *args, **kwargs) -> None:
        self.cls = cls
        self.args = args
        self.kwargs = kwargs

    def __call__(self) -> Any:
        return self.cls(*self.args, **self.kwargs)


@ray.remote
class RayWorker:
    def __init__(self, prefix: str = ""):
        self.model = None
        self.active_key = None

    def init_model(self, cls_with_init):
        self.model = cls_with_init()

    def call_model_method(self, method_name: str, *args, **kwargs):
        method = getattr(self.model, method_name)
        return method(*args, **kwargs)

class Worker:
    def __init__(self, cls_with_init: ClassWithInitArgs):

        self.actor = RayWorker.options(
            num_gpus=len(os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")),
            num_cpus=multiprocessing.cpu_count(),
        ).remote()
        self.cls_with_init = cls_with_init

    def init_worker(self):
        ray.get(self.actor.init_model.remote(self.cls_with_init))

    def __getattribute__(self, name):
        if name in ["actor", "cls_with_init", "init_worker", "kill_worker"]:
            return object.__getattribute__(self, name)
        else:
            def method(*args, **kwargs):
                return ray.get(self.actor.call_model_method.remote(name, *args, **kwargs))
            return method
        
    def __call__(self, *args, **kwargs):
        return ray.get(self.actor.call_model_method.remote("__call__", *args, **kwargs))
    
    def kill_worker(self):
        ray.kill(self.actor)
