import cv2


class RegisterFace(object):

    def __init__(self, sample: int = 100) -> None:
        self.sample = sample
        self.haarCascadePath = "./FaceRecognition/.faces/haarcascade_frontalface_alt2.xml"
        self.captureFaceFromCamera()

    def captureFaceFromCamera(self) -> None:
        i = 0
        capture = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        classifier = cv2.CascadeClassifier(self.haarCascadePath)
        while True:
            ref, frame = capture.read()
            grayFilter = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            regFaceTarget = classifier.detectMultiScale(grayFilter)
            if len(regFaceTarget):
                x, y, w, h = regFaceTarget[0]
                cv2.rectangle(img=frame,
                              pt1=(x, y),
                              pt2=(x + w, y + h),
                              color=(0, 255, 0))
            cv2.imshow("Register Face", frame)
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break
        cv2.destroyAllWindows()


RegisterFace()
