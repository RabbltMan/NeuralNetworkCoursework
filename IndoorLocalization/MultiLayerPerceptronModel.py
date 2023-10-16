from torch import cuda
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
from torch.utils.data import DataLoader
from DataSplit import *
from MultiLayerPerceptronModule import *
from PytorchDatasetGen import *


class MultiLayerPerceptronModel(object):

    def __init__(self, BATCH_SIZE: int = 64, LR: float = 0.01) -> None:
        if cuda.is_available():
            device = "cuda"
            self.model = MultiLayerPerceptronModule().cuda()
        else:
            device = "cpu"
            self.model = MultiLayerPerceptronModule().cpu()
        self.lossFunc = CrossEntropyLoss()
        self.optimizer = Adam(self.model.parameters(), lr=LR)
        Xy = DataSplit()()
        trainingDataset = PytorchDatasetGen(Xy[0], Xy[2], device)
        validationDataset = PytorchDatasetGen(Xy[1], Xy[3], device)
        self.trainingLoader = DataLoader(trainingDataset,
                                         batch_size=BATCH_SIZE,
                                         shuffle=True)
        self.validationLoader = DataLoader(validationDataset,
                                           batch_size=1,
                                           shuffle=False)
        print(f"Model on device: {device}\n{self.model}")

    def train(self, EPOCH: int = 1):
        # TODO(RabbltMan): train loop and per-epoch activity
        pass
