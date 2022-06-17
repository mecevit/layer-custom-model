import layer

class PrivateModel(layer.CustomModel):

    def __init__(self, location: str):
        super().__init__()
        self.location = location
        self.model = None

    def load_model(self):
        import torch
        if self.model is None:
            self.model = torch.load(self.location)
        return self.model