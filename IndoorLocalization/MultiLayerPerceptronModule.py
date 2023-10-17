from torch import Tensor
from torch.nn import Module
from torch.nn import Flatten, Linear, ReLU
from torch.nn.init import normal_


class MultiLayerPerceptronModule(Module):

    def __init__(self, device) -> None:
        super().__init__()
        self.flatten = Flatten()
        self.linear1 = Linear(7, 12, device=device)
        self.linear1.apply(self.init_weights)
        self.relu = ReLU()
        self.linear2 = Linear(12, 4, device=device)
        self.linear2.apply(self.init_weights)

    def init_weights(self, module) -> None:
        normal_(module.weight, std=0.01)

    def forward(self, X) -> Tensor:
        X = self.flatten(X).to(self.linear1.weight.dtype)
        X = self.linear1(X)
        X = self.relu(X)
        X = self.linear2(X)
        return X
