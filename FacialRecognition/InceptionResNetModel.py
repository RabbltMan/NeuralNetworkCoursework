from torch import no_grad, save, load
from torch.nn import TripletMarginLoss, CrossEntropyLoss
from torch.optim import Adam
from torch.utils.data import DataLoader
from os import listdir, path
from InceptionResnetV1Module import *
from TripletDataset import *


class InceptionResNetModel(object):

    def __init__(self, lr: float = 0.00002, BATCH_SIZE: int = 6) -> None:
        self.BATCH_SIZE = BATCH_SIZE
        self.lr = lr
        self.model = InceptionResnetV1Module()
        self.posRootPath = "./FacialRecognition/.faces/"
        self.posPathList = [
            self.posRootPath + d for d in listdir(self.posRootPath)[:-1]
        ]

    def train(self, EPOCH: int = 3, loadPretrained=True):
        self.lossFunc1 = TripletMarginLoss()
        self.lossFunc2 = CrossEntropyLoss()
        self.optimizer = Adam(self.model.parameters(), self.lr)
        bestValLoss = 1_000_000_007
        if loadPretrained and path.exists(
                "./FacialRecognition/InceptionV1.pth"):
            self.model.load_state_dict(
                load("./FacialRecognition/InceptionV1.pth"), assign=True)
            with open("./FacialRecognition/losses.txt", "r") as file:
                bestValLoss = float(file.readlines()[-1].split(",")[-1])
        else:
            with open("./FacialRecognition/losses.txt", "w") as file:
                pass
        for epoch in range(EPOCH):
            # On instantiation, positive(recorded) and negative(strange) faces are pre-loaded(default 500 each).
            # TripletDataLoader itself returns (A, P, N) randomly picked and packed from those faces.
            # There should be P(P(500, 2), 500) different tuples in theory. So val and test shares the same DataLoader.
            # However, this may still cause overfitting with pre-loaded faces.
            # So train set will be updated every epoch.
            self.trainDataset = TripletDataset(self.posPathList, 800)
            self.trainDataLoader = DataLoader(self.trainDataset,
                                              self.BATCH_SIZE)
            trainLen = len(self.trainDataLoader)
            runningLoss = 0
            self.model.train()
            with open("./FacialRecognition/losses.txt", "a") as file:
                for i, (A, P, N, y, yn) in enumerate(self.trainDataLoader):
                    yA = self.model(A)
                    yP = self.model(P)
                    yN = self.model(N)
                    self.optimizer.zero_grad()
                    loss = self.lossFunc1(yA, yP, yN)
                    loss += self.lossFunc2(yA, y)
                    loss += self.lossFunc2(yP, y)
                    loss += 2 * self.lossFunc2(yN, yn)
                    loss.backward()
                    runningLoss += loss.item()
                    print(
                        f"\rEpoch {epoch+1}/{EPOCH}: {(i+1)/trainLen * 100:>6.2f}% >>> loss={runningLoss/(i+1):.4f}",
                        end='')
                    self.optimizer.step()
                    file.write(f"{runningLoss/(i+1)}\n")
            avgTrainLoss = runningLoss / trainLen
            with open("./FacialRecognition/losses.txt", "a") as file:
                file.write(f"{avgTrainLoss}\n")
            if avgTrainLoss < bestValLoss:
                bestValLoss = avgTrainLoss
                save(self.model.state_dict(),
                     f="./FacialRecognition/InceptionV1.pth")
        print()


if __name__ == "__main__":
    newModel = InceptionResNetModel()
    newModel.train()
