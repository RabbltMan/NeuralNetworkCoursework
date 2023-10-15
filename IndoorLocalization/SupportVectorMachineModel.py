from sklearn.svm import SVC
from sklearn.metrics import f1_score, confusion_matrix
from DataSplit import *


class SupportVectorMachineModel(object):

    def __init__(self, C: float = 1.0, kernel='rbf', verbose=True) -> None:
        __X, self.X_test, __y, self.y_test = DataSplit()()
        self.predictor = SVC(C=C, kernel=kernel, verbose=verbose)
        self.predictor.fit(__X, __y)
        self.predict()

    def predict(self):
        y_pred = self.predictor.predict(self.X_test)
        self.f1Score = f1_score(y_true=self.y_test, y_pred=y_pred, average="macro")
        self.confusionMat = confusion_matrix(y_true=self.y_test, y_pred=y_pred)
        print(f"\nF1 Score: {self.f1Score}")
        print(f"Confusion Matrix: \n{self.confusionMat}")
