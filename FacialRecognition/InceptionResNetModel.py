from torch import no_grad, save, load
from torch.nn import TripletMarginLoss, CrossEntropyLoss
from torch.optim import Adam
from torch.utils.data import DataLoader
from os import listdir, path
from InceptionResnetV1Module import *
from TripletDataset import *


class InceptionResNetModel(object):

    def __init__(self, lr: float = 0.0002, BATCH_SIZE: int = 5) -> None:
        self.BATCH_SIZE = BATCH_SIZE
        self.model = InceptionResnetV1Module()
        self.lossFunc1 = TripletMarginLoss()
        self.lossFunc2 = CrossEntropyLoss()
        self.optimizer = Adam(self.model.parameters(), lr)
        self.negPath = "./FacialRecognition/.faces/utkfaces/"
        self.posRootPath = "./FacialRecognition/.faces/"
        self.posPathList = [
            self.posRootPath + d for d in listdir(self.posRootPath)[:-2]
        ]
        self.train()

    def train(self, EPOCH: int = 15, loadPretrained=True):
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
            # So train and test(val) will be updated every 5 epochs.
            if not epoch % 5:
                self.trainTripletDataset = TripletDataset(
                    self.posPathList, self.negPath, 750)
                self.trainTripletDataLoader = DataLoader(
                    self.trainTripletDataset, self.BATCH_SIZE)
                self.testTripletDataset = TripletDataset(
                    self.posPathList, self.negPath, 399)
                self.testTripletDataLoader = DataLoader(
                    self.testTripletDataset, 7)
                trainLen = len(self.trainTripletDataLoader)
            runningLoss = 0
            self.model.train()
            for i, (A, P, N, y) in enumerate(self.trainTripletDataLoader):
                print(
                    f"\rEpoch {epoch+1}/{EPOCH}: {(i+1)/trainLen * 100:>6.2f}%",
                    end='')
                yA = self.model(A)
                yP = self.model(P)
                yN = self.model(N)
                self.optimizer.zero_grad()
                loss = self.lossFunc1(yA, yP, yN) + self.lossFunc2(yA, y)
                loss.backward()
                runningLoss += loss.item()
                self.optimizer.step()
            avgTrainLoss = runningLoss / trainLen
            self.model.eval()
            runningValLoss = 0
            with no_grad():
                for (valA, valP, valN, y) in self.testTripletDataLoader:
                    yValA = self.model(valA)
                    yValP = self.model(valP)
                    yValN = self.model(valN)
                    runningValLoss += self.lossFunc1(yValA, yValP, yValN)
                    runningValLoss += self.lossFunc2(yValA, y)
            avgValLoss = runningValLoss / len(self.testTripletDataLoader)
            print(f" >>> loss={avgTrainLoss:.4f}, val_loss={avgValLoss:.4f}",
                  end='\r')
            with open("./FacialRecognition/losses.txt", "a") as file:
                file.write(f"{avgTrainLoss},{avgValLoss}\n")
            if avgValLoss < bestValLoss:
                bestValLoss = avgValLoss
                save(self.model.state_dict(),
                     f="./FacialRecognition/InceptionV1.pth")
        print()


InceptionResNetModel()
