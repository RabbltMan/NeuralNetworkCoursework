import cv2
from torch import cat, cuda, load
from torchvision.transforms import ToTensor
import json
from random import random
from time import sleep
from threading import Thread
from InceptionResNetModel import InceptionResNetModel
from StrangerEnhancement import StrangerEnhancement


class FacialRecognition(object):

    def __init__(self) -> None:
        self.resnet = InceptionResNetModel()
        self.resnet.model.load_state_dict(
            load("./FacialRecognition/InceptionV1.pth"), assign=True)
        self.resnet.model.eval()
        self.snapshots = []
        self.snapshotLock: bool = False
        self.haarCascadePath = "./FacialRecognition/.faces/z_haarcascade_frontalface_alt2.xml"
        self.markFace()

    def markFace(self) -> None:
        capture = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        classifier = cv2.CascadeClassifier(self.haarCascadePath)
        while (cv2.waitKey(1) != ord("q")):
            _, frame = capture.read()
            ContinuousNoTargetFrame = 0
            grayFilter = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            regFaceTarget = classifier.detectMultiScale(grayFilter)
            if len(regFaceTarget):
                ContinuousNoTargetFrame = 0
                x, y, w, h = regFaceTarget[0]
                cv2.rectangle(img=frame,
                              pt1=(x, y - 10),
                              pt2=(x + w, y + h + 10),
                              color=(0, 255, 0))
                face = cv2.resize(frame[y:y + h, x + 2:x - 2 + w], (160, 160))
                if not self.snapshotLock and random() > 0.6:
                    self.snapshots.append(ToTensor()(face).unsqueeze(0))
                    self.snapshots.append(face)
                    self.snapshotLock = len(self.snapshots) == 6
                    if self.snapshotLock:
                        recognitThread = Thread(target=self.recognizeFace,
                                                name="recognit")
                        recognitThread.start()
            else:
                ContinuousNoTargetFrame += 1
                if (ContinuousNoTargetFrame >= 30):
                    self.snapshots = []
                    ContinuousNoTargetFrame = 0
            cv2.imshow(f"Face Recognition", frame)

    def recognizeFace(self):
        with open("./FacialRecognition/users.json", "r") as f:
            registeredUsers = json.load(f)
        imgs, npImgs = cat(self.snapshots[::2]), self.snapshots[1::2]
        se = StrangerEnhancement()
        if all([se.predict(npImgs[i]) for i in range(3)]):
            print("Unknown")
            sleep(1)
            self.snapshots = []
            self.snapshotLock = False
        else:
            if cuda.is_available():
                print(self.resnet.model(imgs.cuda()))
                y = [
                    i.item()
                    for i in self.resnet.model(imgs.cuda()).cpu().argmax(dim=1)
                ]
                print(registeredUsers[str(max(y, key=y.count))])
                sleep(1)
                self.snapshots = []
                self.snapshotLock = False
            else:
                self.resnet.model(imgs.cpu())
                y = [
                    i.item()
                    for i in self.resnet.model(imgs.cpu()).argmax(dim=1)
                ]
                print(registeredUsers[str(max(y, key=y.count))])
                sleep(1)
                self.snapshots = []
                self.snapshotLock = False
        


if __name__ == "__main__":
    FacialRecognition()
