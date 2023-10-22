from torch import Tensor, cat, cuda
from torch.nn import *
from os import listdir
from typing import Tuple, Union


class InceptionResnetV1Module(Module):

    def __init__(
        self,
        classNum: int = len(listdir("./FacialRecognition/.faces/")) - 2
    ) -> None:
        super().__init__()
        self.conv2d_1a = Conv2dStack(3, 32, 3, 2)
        self.conv2d_2a = Conv2dStack(32, 32, 3, 1)
        self.conv2d_2b = Conv2dStack(32, 64, 3, 1, 1)
        self.maxPool_3a = MaxPool2d(3, 2)
        self.conv2d_3b = Conv2dStack(64, 80, 1, 1)
        self.conv2d_4a = Conv2dStack(80, 192, 3, 1)
        self.conv2d_4b = Conv2dStack(192, 256, 3, 2)
        self.blockStack_1 = Sequential(Block35(0.17), Block35(0.17),
                                       Block35(0.17), Block35(0.17),
                                       Block35(0.17))
        self.mixed_6a = Mixed_6a()
        self.blockStack_2 = Sequential(
            Block17(0.1),
            Block17(0.1),
            Block17(0.1),
            Block17(0.1),
            Block17(0.1),
            Block17(0.1),
            Block17(0.1),
            Block17(0.1),
            Block17(0.1),
            Block17(0.1),
        )
        self.mixed_7a = Mixed_7a()
        self.blockStack_3 = Sequential(Block8(0.2), Block8(0.2), Block8(0.2),
                                       Block8(0.2), Block8(0.2))
        self.block8 = Block8(noReLU=True)
        self.avgPool_1a = AdaptiveAvgPool2d(1)
        self.dropout = Dropout(0.6)
        self.linear = Linear(1792, 512, False)
        self.bn = BatchNorm1d(512, 0.001)
        self.logits = Linear(512, classNum)
        if cuda.is_available():
            self.cuda()
            print(f"Model on device: cuda")
        else:
            self.cpu()
            print(f"Model on device: cpu")

    def forward(self, X: Tensor) -> Tensor:
        X = self.conv2d_1a(X)
        X = self.conv2d_2a(X)
        X = self.conv2d_2b(X)
        X = self.maxPool_3a(X)
        X = self.conv2d_3b(X)
        X = self.conv2d_4a(X)
        X = self.conv2d_4b(X)
        X = self.blockStack_1(X)
        X = self.mixed_6a(X)
        X = self.blockStack_2(X)
        X = self.mixed_7a(X)
        X = self.blockStack_3(X)
        X = self.block8(X)
        X = self.avgPool_1a(X)
        X = self.dropout(X)
        X = self.linear(X.view(X.shape[0], -1))
        X = self.bn(X)
        X = self.logits(X)
        return X


class Conv2dStack(Module):

    def __init__(self,
                 inChannels: int,
                 outChannels: int,
                 kernelSize: Union[int, Tuple[int, int]],
                 stride: Union[int, Tuple[int, int]],
                 padding: Union[int, Tuple[int, int]] = 0) -> None:
        super().__init__()
        self.conv2d = Conv2d(inChannels,
                             outChannels,
                             kernelSize,
                             stride,
                             padding,
                             bias=False)
        self.bn = BatchNorm2d(outChannels, eps=0.001)
        self.relu = ReLU()

    def forward(self, X) -> Tensor:
        X = self.conv2d(X)
        X = self.bn(X)
        X = self.relu(X)
        return X


class Block35(Module):

    def __init__(self, scale: float = 1.0) -> None:
        super().__init__()
        self.scale = scale
        self.branch_0 = Conv2dStack(256, 32, 1, 1)
        self.branch_1 = Sequential(Conv2dStack(256, 32, 1, 1),
                                   Conv2dStack(32, 32, 3, 1, 1))
        self.branch_2 = Sequential(Conv2dStack(256, 32, 1, 1),
                                   Conv2dStack(32, 32, 3, 1, 1),
                                   Conv2dStack(32, 32, 3, 1, 1))
        self.conv2d = Conv2d(96, 256, 1)
        self.relu = ReLU()

    def forward(self, X: Tensor) -> Tensor:
        X_0 = self.branch_0(X)
        X_1 = self.branch_1(X)
        X_2 = self.branch_2(X)
        y = cat((X_0, X_1, X_2), 1)
        y = self.conv2d(y)
        y = y * self.scale + X
        y = self.relu(y)
        return y


class Block17(Module):

    def __init__(self, scale: float = 1.0) -> None:
        super().__init__()
        self.scale = scale
        self.branch_0 = Conv2dStack(896, 128, 1, 1)
        self.branch_1 = Sequential(Conv2dStack(896, 128, 1, 1),
                                   Conv2dStack(128, 128, (1, 7), 1, (0, 3)),
                                   Conv2dStack(128, 128, (7, 1), 1, (3, 0)))
        self.conv2d = Conv2d(256, 896, 1, 1)
        self.relu = ReLU()

    def forward(self, X: Tensor) -> Tensor:
        X_0 = self.branch_0(X)
        X_1 = self.branch_1(X)
        y = cat((X_0, X_1), 1)
        y = self.conv2d(y)
        y = y * self.scale + X
        y = self.relu(y)
        return y


class Block8(Module):

    def __init__(self, scale: float = 1.0, noReLU: bool = False):
        super().__init__()
        self.scale = scale
        self.noReLU = noReLU
        self.branch_0 = Conv2dStack(1792, 192, 1, 1)
        self.branch_1 = Sequential(Conv2dStack(1792, 192, 1, 1),
                                   Conv2dStack(192, 192, (1, 3), 1, (0, 1)),
                                   Conv2dStack(192, 192, (3, 1), 1, (1, 0)))
        self.conv2d = Conv2d(384, 1792, 1, 1)
        if not self.noReLU:
            self.relu = ReLU()

    def forward(self, X):
        X_0 = self.branch_0(X)
        X_1 = self.branch_1(X)
        y = cat((X_0, X_1), 1)
        y = self.conv2d(y)
        y = y * self.scale + X
        if not self.noReLU:
            y = self.relu(y)
        return y


class Mixed_6a(Module):

    def __init__(self) -> None:
        super().__init__()
        self.branch_0 = Conv2dStack(256, 384, 3, 2)
        self.branch_1 = Sequential(Conv2dStack(256, 192, 1, 1),
                                   Conv2dStack(192, 192, 3, 1, 1),
                                   Conv2dStack(192, 256, 3, 2))
        self.branch_2 = MaxPool2d(3, 2)

    def forward(self, X: Tensor) -> Tensor:
        X_0 = self.branch_0(X)
        X_1 = self.branch_1(X)
        X_2 = self.branch_2(X)
        y = cat((X_0, X_1, X_2), 1)
        return y


class Mixed_7a(Module):

    def __init__(self) -> None:
        super().__init__()
        self.branch_0 = Sequential(Conv2dStack(896, 256, 1, 1),
                                   Conv2dStack(256, 384, 3, 2))
        self.branch_1 = Sequential(Conv2dStack(896, 256, 1, 1),
                                   Conv2dStack(256, 256, 3, 2))
        self.branch_2 = Sequential(Conv2dStack(896, 256, 1, 1),
                                   Conv2dStack(256, 256, 3, 1, 1),
                                   Conv2dStack(256, 256, 3, 2))
        self.branch_3 = MaxPool2d(3, 2)

    def forward(self, X: Tensor) -> Tensor:
        X_0 = self.branch_0(X)
        X_1 = self.branch_1(X)
        X_2 = self.branch_2(X)
        X_3 = self.branch_3(X)
        y = cat((X_0, X_1, X_2, X_3), 1)
        return y
