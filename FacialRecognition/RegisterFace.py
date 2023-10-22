import cv2
import os
from PIL import Image, ImageEnhance
from random import random, choice


class RegisterFace(object):

    def __init__(self, sample: int = 50, augmentNum: int = 20) -> None:
        self.augmentNum = augmentNum
        self.sample = sample
        self.haarCascadePath = "./FacialRecognition/.faces/haarcascade_frontalface_alt2.xml"
        self.captureFaceFromCamera()

    def captureFaceFromCamera(self) -> None:
        regNum = len(os.listdir("./FacialRecognition/.faces/")) - 1
        self.path = f"./FacialRecognition/.faces/{regNum}/"
        os.mkdir(f"{self.path}")
        i = 0
        capture = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        classifier = cv2.CascadeClassifier(self.haarCascadePath)
        while (i < self.sample):
            ref, frame = capture.read()
            grayFilter = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            regFaceTarget = classifier.detectMultiScale(grayFilter)
            if len(regFaceTarget):
                x, y, w, h = regFaceTarget[0]
                cv2.rectangle(img=frame,
                              pt1=(x, y - 10),
                              pt2=(x + w, y + h + 10),
                              color=(0, 255, 0))
                # press SPACE to sample
                if cv2.waitKey(1) == 32:
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
            cv2.imshow(f"Register New Face#{regNum}", frame)
            # press Q to quit
            if cv2.waitKey(1) & 0xFF == ord("q"):
                if not len(os.listdir(self.path)):
                    os.rmdir(self.path)
                break
        cv2.destroyAllWindows()

    def randAugment(self, img, i, j):
        enhancement = 0
        while (enhancement < 3 and (random()**enhancement) > 0.5):
            factor = 1.5 * random() + 0.35
            enhancer = choice([
                ImageEnhance.Brightness(img),
                ImageEnhance.Contrast(img),
                ImageEnhance.Color(img),
                ImageEnhance.Sharpness(img)
            ])
            img = enhancer.enhance(factor)
            enhancement += 1
            img.save(f"{self.path}{i}_{j}.png")


RegisterFace()
