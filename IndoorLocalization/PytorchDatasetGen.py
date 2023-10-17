from typing import Literal, Tuple
from torch import from_numpy, Tensor
from torch.utils.data import Dataset
from DataSplit import *


class PytorchDatasetGen(Dataset):

    def __init__(self,
                 X: ndarray,
                 y: ndarray,
                 device: Literal["cuda", "cpu"] = "cpu") -> None:
        self.__len = X.shape[0]
        self.X = from_numpy(X).to(device)
        self.y = from_numpy(y).add(-1).squeeze().long().to(device)

    def __getitem__(self, index: int) -> Tuple[Tensor, Tensor]:
        return self.X[index], self.y[index]

    def __len__(self) -> int:
        return self.__len
