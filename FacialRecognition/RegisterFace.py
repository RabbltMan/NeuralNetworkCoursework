import cv2
import os
from PIL import Image, ImageEnhance
from random import random, choice


class RegisterFace(object):

    def __init__(self, sample: int = 30, augmentNum: int = 30) -> None:
        self.augmentNum = augmentNum
        self.sample = sample
        self.haarCascadePath = "./FacialRecognition/.faces/z_haarcascade_frontalface_alt2.xml"
        self.captureFaceFromCamera()

    def captureFaceFromCamera(self) -> None:
        self.regNum = len(os.listdir("./FacialRecognition/.faces/")) - 1
        self.path = f"./FacialRecognition/.faces/{self.regNum}/"
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
            cv2.imshow(f"Register New Face#{self.regNum}", frame)
            # press Q to quit
            if cv2.waitKey(1) & 0xFF == ord("q"):
                if not len(os.listdir(self.path)):
                    os.rmdir(self.path)
                break
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
        img.save(f"{self.path}{self.regNum}_{i}_{j}.png")


RegisterFace()
