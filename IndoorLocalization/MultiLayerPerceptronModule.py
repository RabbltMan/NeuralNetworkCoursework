from torch import Tensor
from torch.nn import Module, Sequential
from torch.nn import Flatten, Linear, ReLU
from torch.nn.init import normal_


class MultiLayerPerceptronModule(Module):

    def __init__(self) -> None:
        super().__init__()
        self.flatten = Flatten()
        self.linear_relu_stack = Sequential(Linear(7, 6), ReLU(), Linear(6, 4))
        self.linear_relu_stack.apply(self.init_weights)

    def init_weights(self, module) -> None:
        if isinstance(module, Linear):
            normal_(module.weight, std=0.01)

    def forward(self, X) -> Tensor:
        X = self.flatten(X)
        y = self.linear_relu_stack(X)
        return y
