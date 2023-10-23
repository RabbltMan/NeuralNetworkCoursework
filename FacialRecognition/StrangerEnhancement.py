import cv2
from numpy import array
from torch import Tensor
import json
from os import listdir
from random import choices


class StrangerEnhancement(object):

    def __init__(self) -> None:
        pass

    @staticmethod
    def train():
        src = []
        with open("./FacialRecognition/users.json", 'r') as f:
            labels = json.load(f).keys()
        for label in labels:
            path = f"./FacialRecognition/.faces/{label}/"
            src += [
                cv2.imread(f"{path}/{f}", cv2.IMREAD_GRAYSCALE)
                for f in choices(listdir(path), k=16)
            ]
        labels = array([int(key) for key in labels for _ in range(16)])
        strangerRecogn = cv2.face.LBPHFaceRecognizer_create()
        strangerRecogn.train(src, labels)
        strangerRecogn.save("./FacialRecognition/stranger.yaml")

    def predict(self, img) -> bool:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        strangerRecogn = cv2.face.LBPHFaceRecognizer_create()
        strangerRecogn.read("./FacialRecognition/stranger.yaml")
        y, minError = strangerRecogn.predict(img)
        print(y, minError)
        if round(minError) > 80:
            return True
        return False


if __name__ == "__main__":
    StrangerEnhancement().train()
