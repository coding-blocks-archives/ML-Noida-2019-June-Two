import cv2
import numpy as np
from sklearn.neighbors import KNeighborsClassifier


class KNN:

    def __init__(self, k):
        self.k = k
        self.X = None
        self.y = None

    def fit(self, X, y):
        self.X = X
        self.y = y

    def predict_point(self, point):

        distances = []
        for x_item, y_item in zip(self.X, self.y):
            dis = KNN.distance(x_item, point)
            distances.append((dis, y_item))

        distances = sorted(distances)
        top = np.array(distances[:self.k])

        vals, counts = np.unique(top[:, 1], return_counts=True)
        return vals[np.argmax(counts)]

    def predict(self, X):
        res = []
        for point in X:
            v = self.predict_point(point)
            res.append(v)
        return np.array(res)

    def score(self, X, y):
        return sum(self.predict(X) == y) / len(y)

    @classmethod
    def distance(cls, p1, p2):
        return np.sum((p1 - p2) ** 2)

cap = cv2.VideoCapture(0)
classifier = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

data = np.load("face_data.npy")
X = data[:, 1:].astype(int)
y = data[:, 0]

model = KNN(5)
model.fit(X, y)

while True:
    ret, frame = cap.read()

    if ret:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = classifier.detectMultiScale(gray)

        areas = []
        for face in faces:
            x, y, w, h = face
            areas.append((w*h, face))

        if len(faces) > 0:
            face = max(areas)[1]
            x, y, w, h = face

            face_img = gray[y:y+h, x:x+w]

            face_img = cv2.resize(face_img, (100, 100))
            face_flatten = face_img.flatten()

            res = model.predict([face_flatten])
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 255, 255), 4)
            cv2.putText(frame, str(res[0]), (x + 20, y - 20), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255))


        cv2.imshow("video", frame)

    key = cv2.waitKey(1)
    if key & 0xff == ord('q'):
        break
