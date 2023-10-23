import cv2
from PIL import Image, ImageEnhance
import json
import os
from random import random, choice
from time import time
from StrangerEnhancement import *


class RegisterFace(object):

    def __init__(self, user: str = f"user{int(time())}", sample: int = 30, augmentNum: int = 30) -> None:
        self.user = user
        self.augmentNum = augmentNum
        self.sample = sample
        self.haarCascadePath = "./FacialRecognition/.faces/z_haarcascade_frontalface_alt2.xml"
        self.captureFaceFromCamera()

    def initJson(self, regId: int):
        users = dict()
        jsonPath = "./FacialRecognition/users.json"
        if not os.path.exists(jsonPath):
            with open(jsonPath, "w", encoding="utf-8") as f:
                users[regId] = self.user
                json.dump(users, f)
        else:
            with open(jsonPath, "r", encoding="utf-8") as f:
                users = json.load(f)
            with open(jsonPath, "w", encoding="utf-8") as f:
                users[regId] = self.user
                json.dump(users, f)

    def captureFaceFromCamera(self) -> None:
        self.regId = len(os.listdir("./FacialRecognition/.faces/")) - 2
        self.path = f"./FacialRecognition/.faces/{self.regId}/"
        self.initJson(self.regId)
        os.mkdir(f"{self.path}")
        i = 0
        capture = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        classifier = cv2.CascadeClassifier(self.haarCascadePath)
        while (i < self.sample):
            _, frame = capture.read()
            grayFilter = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            regFaceTarget = classifier.detectMultiScale(grayFilter)
            if len(regFaceTarget):
                x, y, w, h = regFaceTarget[0]
                cv2.rectangle(img=frame,
                              pt1=(x, y - 10),
                              pt2=(x + w, y + h + 10),
                              color=(0, 255, 0))
                # press SPACE to sample
                if random() > 0.7:
                    aug = 0
                    face = frame[y:y + h, x + 2:x - 2 + w]
                    cv2.imwrite(f"{self.path}{i}_{aug}.png", face)
                    img = Image.open(f"{self.path}{i}_{aug}.png")
                    img = img.resize((160, 160))
                    img.save(f"{self.path}{i}_{aug}.png")
                    while (aug < self.augmentNum):
                        self.randAugment(img, i, aug)
                        aug += 1
                    i += 1
            cv2.imshow(f"Register New Face#{self.regId}", frame)
            # press Q to quit
            if cv2.waitKey(1) & 0xFF == ord("q"):
                if not len(os.listdir(self.path)):
                    os.rmdir(self.path)
                break
        StrangerEnhancement.train()
        cv2.destroyAllWindows()

    def randAugment(self, img: Image.Image, i, j):
        enhancement = 0
        if (random() > 0.5):
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
        while (enhancement < 3 and (random() > 0.5 or enhancement == 0)):
            factor = 0.5 * random() + 0.75
            enhancer = choice([
                ImageEnhance.Brightness(img),
                ImageEnhance.Contrast(img),
                ImageEnhance.Color(img),
                ImageEnhance.Sharpness(img),
            ])
            img = enhancer.enhance(factor)
            enhancement += 1
        img.save(f"{self.path}{self.regId}_{i}_{j}.png")

if __name__ == "__main__":
    RegisterFace("Guo")