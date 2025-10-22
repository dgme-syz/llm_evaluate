# comet_worker.py
import sys
import json

import torch
from comet import download_model, load_from_checkpoint
from comet.models.utils import Prediction
from torch.serialization import safe_globals

data_path = sys.argv[1]
model_name = sys.argv[2]
output_file = sys.argv[3] 

with open(data_path, "r") as f:
    data = json.load(f)
torch.serialization.add_safe_globals([Prediction])
gpus = torch.cuda.device_count()
with safe_globals([Prediction]):
    
    model_path = download_model(model_name)
    comet_model = load_from_checkpoint(model_path, local_files_only=False)
    prediction = comet_model.predict(data, batch_size=64, gpus=gpus)    

outputs: dict = {"extra_dict": {}}
if hasattr(prediction, "system_score"):
    outputs["score"] = float(prediction.system_score)
if hasattr(prediction, "scores"):
    outputs["extra_dict"]["score_per_example"] = list(prediction.scores)

with open(output_file, "w") as f:
    json.dump(outputs, f)
