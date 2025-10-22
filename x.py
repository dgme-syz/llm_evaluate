
from comet import download_model, load_from_checkpoint

from comet.models.utils import Prediction
import torch
torch.serialization.add_safe_globals([Prediction])


print("Comet Worker Test >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
if __name__ == '__main__':
        model_path = download_model("Unbabel/wmt22-comet-da")
        # or for example:
        # model_path = download_model("Unbabel/wmt22-comet-da")

        # Load the model checkpoint:
        model = load_from_checkpoint(model_path)

        # Data must be in the following format:
        data = [
                {
                                "src": "10 到 15 分钟可以送到吗",
                                        "mt": "Can I receive my food in 10 to 15 minutes?",
                                                "ref": "Can it be delivered between 10 to 15 minutes?"
                                                },
                        {
                                "src": "Pode ser entregue dentro de 10 a 15 minutos?",
                                        "mt": "Can you send it for 10 to 15 minutes?",
                                                "ref": "Can it be delivered between 10 to 15 minutes?"
                                                        }
                        ]
        # Call predict method:

        model_output = model.predict(data, batch_size=8, gpus=2)
        print(model_output)
