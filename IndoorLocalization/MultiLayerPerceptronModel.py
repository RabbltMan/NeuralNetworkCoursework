from pandas import DataFrame
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, confusion_matrix
from torch import cuda, no_grad
from torch import save, load, argmax
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
from torch.utils.data import DataLoader
from DataSplit import *
from MultiLayerPerceptronModule import *
from PytorchDatasetGen import *


class MultiLayerPerceptronModel(object):

    def __init__(self, BATCH_SIZE: int = 64, LR: float = 0.01) -> None:
        self.BATCHSIZE = BATCH_SIZE
        self.LR = LR
        if cuda.is_available():
            self.__device = "cuda"
            self.model = MultiLayerPerceptronModule(self.__device).cuda()
        else:
            self.__device = "cpu"
            self.model = MultiLayerPerceptronModule(self.__device).cpu()
        self.lossFunc = CrossEntropyLoss()
        self.optimizer = Adam(self.model.parameters(), lr=self.LR)
        self.Xy = DataSplit()()
        self.testSet = PytorchDatasetGen(self.Xy[1], self.Xy[3], self.__device)
        self.testLoader = DataLoader(self.testSet,
                                     batch_size=len(self.testSet),
                                     shuffle=True)
        print(f"Model on device: {self.__device}\n{self.model}\n")
        self.train(150)
        self.evaluate()

    def train(self, EPOCHS: int = 1):
        self.EPOCHS = EPOCHS
        # Split training data -> 0.9 Training, 0.1 Validation
        SKFold = list(StratifiedKFold(10).split(self.Xy[0], self.Xy[2]))
        currEpoch, bestValLoss = 0, 1_000_000_007
        for _ in range(EPOCHS):
            if not (currEpoch % 5):
                Xi, val_Xi = SKFold[(currEpoch // 5) % 10]
                self.trainSet = (DataFrame(self.Xy[0]).iloc[Xi].to_numpy(),
                                 DataFrame(self.Xy[2]).iloc[Xi].to_numpy())
                self.valSet = (DataFrame(self.Xy[0]).iloc[val_Xi].to_numpy(),
                               DataFrame(self.Xy[2]).iloc[val_Xi].to_numpy())
                self.trainSet = PytorchDatasetGen(*self.trainSet,
                                                  self.__device)
                self.trainLoader = DataLoader(self.trainSet,
                                              batch_size=self.BATCHSIZE,
                                              shuffle=True)
                self.valSet = PytorchDatasetGen(*self.valSet, self.__device)
                self.valLoader = DataLoader(self.valSet,
                                            batch_size=len(self.valSet),
                                            shuffle=True)
            self.model.train(True)
            avgTrainLoss = self.__trainEpoch(currEpoch)
            # Validation
            runningValLoss = 0.0
            self.model.eval()
            with no_grad():
                for valData in self.valLoader:
                    val_X, val_y = valData
                    val_y_pred = self.model(val_X)
                    valLoss = self.lossFunc(val_y_pred, val_y)
                    runningValLoss += valLoss
            avgValLoss = runningValLoss / len(self.valLoader)
            print(f" >>> loss={avgTrainLoss:.4f}, val_loss={avgValLoss:.4f}",
                  end='\r')

            if avgValLoss < bestValLoss:
                bestValLoss = avgValLoss
                save(self.model.state_dict(),
                     f="./IndoorLocalization/MLP_checkpoint.pt")
            currEpoch += 1
        print()

    def __trainEpoch(self, epoch) -> float:
        runningLoss: float = 0.0
        for i, data in enumerate(self.trainLoader):
            print(
                f"\rEpoch {epoch+1}/{self.EPOCHS}: {(i+1)/len(self.trainLoader)*100:.2f}%",
                end='')
            # Track data batch from dataloader
            X, y = data
            # Zero grad for every batch
            self.optimizer.zero_grad()
            # Compute loss and grad
            y_pred = self.model(X)
            loss = self.lossFunc(y_pred, y)
            loss.backward()
            runningLoss += loss.item()
            # Adjust learning weights
            self.optimizer.step()

        return runningLoss / len(self.trainLoader)

    def evaluate(self):
        savedState = load("./IndoorLocalization/MLP_checkpoint.pt", mmap=True)
        net = MultiLayerPerceptronModule(self.__device)
        net.load_state_dict(savedState, assign=True)
        runningTestLoss = 0.0
        net.eval()
        with no_grad():
            for i, testData in enumerate(self.testLoader):
                print(f"\rTest Set: {(i+1)/len(self.testLoader)*100:.2f}%",
                      end='')
                test_X, test_y = testData
                test_y_pred = self.model(test_X)
                testLoss = self.lossFunc(test_y_pred, test_y)
                runningTestLoss += testLoss
        avgTestLoss = runningTestLoss / len(self.testLoader)
        f1Score = f1_score(test_y.cpu(),
                           argmax(test_y_pred.cpu(), dim=1),
                           average="macro")
        confMat = confusion_matrix(test_y.cpu(),
                                   argmax(test_y_pred.cpu(), dim=1))
        print(f"\n\nPerformance: \ntest_loss: {avgTestLoss:.4f}")
        print(f"F1 Score: {f1Score}")
        print(f"Confusion Matrix: \n{confMat}")
