import layer
import torch.nn
from layer.decorators import model
import os
from private_model import PrivateModel


@model("private_model")
def train():
    my_private_model = torch.nn.Sequential()
    path = "model.pt"
    torch.save(my_private_model, path)
    return PrivateModel(path)


if __name__ == "__main__":
    layer_api_key = os.environ['LAYER_API_KEY']  # Get your api key at https://app.layer.ai/me/settings/developer
    layer.login_with_api_key(layer_api_key)
    layer.init("custom-model")
    # train()

    # Get model
    layer_model = layer.get_model("private_model:2.1").get_train()
    my_private_model = layer_model.load_model()
    print(my_private_model)
